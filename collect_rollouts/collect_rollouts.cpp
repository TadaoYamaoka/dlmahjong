#include "../../cmajiang/src_cpp/game.h"
#include "../../cmajiang/src_cpp/random.h"
#include "../../cmajiang/src_cpp/paipu.h"

#include "inference.h"

#include "cxxopts.hpp"

#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <filesystem>
#include <exception>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <functional>

std::atomic<bool> stop = false;

std::string get_current_datetime() {
    // 現在の時刻を取得
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
    localtime_s(&tm_now, &time_t_now);

    // 文字列ストリームを使用してフォーマット
    std::stringstream ss;
    ss << std::put_time(&tm_now, "%Y%m%d%H%M%S");

    return ss.str();
}

// 牌譜
std::vector<Game::Paipu> paipu;
std::mutex paipu_mutex;

struct ActionReply {
    Action action;
    Game::Reply reply;
};

std::vector<ActionReply> get_leagal_actions(const Game& game, const int player_id) {
    std::vector<ActionReply> leagal_actions;

    switch (game.status()) {
    case Game::Status::ZIMO:
    case Game::Status::GANGZIMO:
        if (game.allow_hule()) {
            // 和了
            leagal_actions.emplace_back(ActionReply{ Action::HULE, Game::Reply{ Game::Message::HULE, {} } });
        }
        // 打牌
        for (const auto& p : game.get_dapai()) {
            if (p.back() == '_') {
                leagal_actions.emplace_back(ActionReply{ Action::DAPAI_ZIMO, Game::Reply{ Game::Message::DAPAI, p } });
            }
            else {
                const int i = index_of(p[0]) * 10 + (p[1] == '0' ? 9 : p[1] - '1');
                leagal_actions.emplace_back(ActionReply{ Action::DAPAI_M1 + i, Game::Reply{ Game::Message::DAPAI, p } });
            }
        }
        // 立直
        for (const auto& p : game.allow_lizhi().second) {
            if (p.back() == '_') {
                leagal_actions.emplace_back(ActionReply{ Action::LIZHI_ZIMO, Game::Reply{ Game::Message::DAPAI, p + '*' } });
            }
            else {
                const int i = index_of(p[0]) * 10 + (p[1] == '0' ? 9 : p[1] - '1');
                leagal_actions.emplace_back(ActionReply{ Action::LIZHI_M1 + i, Game::Reply{ Game::Message::DAPAI, p + '*' } });
            }
        }
        // 暗槓もしくは加槓
        for (const auto& m : game.get_gang_mianzi()) {
            const int i = index_of(m[0]) * 10 + (m[1] == '0' ? 4 : m[1] - '1');
            leagal_actions.emplace_back(ActionReply{ Action::GANG_M1 + i, Game::Reply{ Game::Message::GANG, m } });
        }
        break;
    case Game::Status::DAPAI:
    {
        // 他家の応答 ロン、副露
        const auto player_lunban = game.player_lunban(player_id);
        if (game.allow_hule(player_lunban)) {
            // ロン
            leagal_actions.emplace_back(ActionReply{ Action::HULE, Game::Reply{ Game::Message::HULE, {} } });
        }
        // 副露
        // チー
        for (const auto& m : game.get_chi_mianzi(player_lunban)) {
            const int i = m[2] == '-' ? (m[3] == '0' || m[4] == '0' ? 3 : 0) : m[3] == '-' ? (m[1] == '0' || m[4] == '0' ? 4 : 1) : (m[1] == '0' || m[2] == '0' ? 5 : 2);
            leagal_actions.emplace_back(ActionReply{ Action::CHI_L + i, Game::Reply{ Game::Message::FULOU, m } });
        }
        // ポン
        for (const auto& m : game.get_peng_mianzi(player_lunban)) {
            leagal_actions.emplace_back(ActionReply{ m[1] == '0' || m[2] == '0' ? Action::PENG_H : Action::PENG, Game::Reply{ Game::Message::FULOU, m } });
        }
        // 明槓
        for (const auto& m : game.get_gang_mianzi(player_lunban)) {
            leagal_actions.emplace_back(ActionReply{ Action::GANG, Game::Reply{ Game::Message::GANG, m } });
        }
        // 鳴かない、ロンしない
        if (leagal_actions.size() > 0) {
            leagal_actions.emplace_back(ActionReply{ Action::NO_ACTION, Game::Reply{} });
        }
        break;
    }
    case Game::Status::FULOU:
    {
        for (const auto& p : game.get_dapai()) {
            const int i = index_of(p[0]) * 10 + (p[1] == '0' ? 9 : p[1] - '1');
            leagal_actions.emplace_back(ActionReply{ Action::DAPAI_M1 + i, Game::Reply{ Game::Message::DAPAI, p } });
        }
        break;
    }
    case Game::Status::GANG:
    {
        const int player_lunban = game.player_lunban(player_id);
        if (game.allow_hule(player_lunban)) {
            // ロン(槍槓)
            leagal_actions.emplace_back(ActionReply{ Action::HULE, Game::Reply{ Game::Message::HULE, {} } });
            // ロンしない
            leagal_actions.emplace_back(ActionReply{ Action::NO_ACTION, Game::Reply{} });
        }
        break;
    }
    }
    return leagal_actions;
}

class PolicyInference : public Inference {
public:
    PolicyInference(const char* filepath, const int max_batch_size) : Inference{ filepath, max_batch_size} {}

    void forward(const int batch_size, public_features_t* features_batch, policy_t* policy_batch) {
        std::lock_guard<std::mutex> lock(mutex);
        Inference::forward(batch_size, (float*)features_batch, (float*)policy_batch);
    }

    int get_max_batch_size() const { return max_batch_size; }

private:
    std::mutex mutex;
};

void collect_rollouts(PolicyInference& inference, const int n_games) {
    std::vector<Game> games(n_games);
    public_features_t* features_batch;
    policy_t* policy_batch;
    const auto max_batch_size = inference.get_max_batch_size();
    checkCudaErrors(cudaHostAlloc((void**)&features_batch, sizeof(public_features_t) * max_batch_size, cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc((void**)&policy_batch, sizeof(policy_t) * max_batch_size, cudaHostAllocPortable));

    struct GamePlayerIdTbl {
        int game_id;
        int player_id;
        std::vector<ActionReply> leagal_actions;
    };
    std::vector<GamePlayerIdTbl> game_player_id_tbl;

    std::random_device seed_gen;
    std::mt19937_64 mt(seed_gen());

    const int n_xiangting = 3;

    while (!stop) {
        game_player_id_tbl.clear();

        public_features_t* features = features_batch;
        for (int game_id = 0; game_id < n_games; game_id++) {
            auto& game = games[game_id];

            // ゲーム初期化
            if (game.status() == Game::Status::NONE || game.status() == Game::Status::JIEJI) {
                game.kaiju();
                random_game_state(game, n_xiangting, mt);
            }
            else if (game.status() == Game::Status::QIPAI) {
                random_game_state(game, n_xiangting, mt);
            }

            for (int player_id = 0; player_id < 4; player_id++) {
                // 合法手
                if (player_id == game.lunban_player_id()) {
                    if (game.status() == Game::Status::DAPAI || game.status() == Game::Status::GANG)
                        continue;
                }
                else {
                    if (game.status() != Game::Status::DAPAI && game.status() != Game::Status::GANG)
                        continue;
                }
                auto leagal_actions = get_leagal_actions(game, player_id);
                if (leagal_actions.size() == 0)
                    continue;
                game_player_id_tbl.emplace_back(GamePlayerIdTbl{ game_id, player_id, std::move(leagal_actions) });

                // 特徴量作成
                const auto lunban = game.lunban();
                std::fill_n((float*)features, sizeof(public_features_t) / sizeof(float), 0);
                public_features(game, lunban, (channel_t*)features);
                // player id
                fill_channel(&(*features)[N_CHANNELS_PUBLIC + player_id], 1);

                features++;
            }

        }

        if (game_player_id_tbl.size() > 0) {
            // 推論(並列実行しているゲームの全プレイヤー分をまとめて推論)
            inference.forward((const int)game_player_id_tbl.size(), features_batch, policy_batch);

            // 推論結果のpolicyからサンプリングして行動
            policy_t* policy = policy_batch;
            for (const auto& game_player_id : game_player_id_tbl) {
                const int game_id = game_player_id.game_id;
                const int player_id = game_player_id.player_id;
                const auto& leagal_actions = game_player_id.leagal_actions;

                auto& game = games[game_id];

                // 合法手でフィルター
                std::vector<double> probability;
                probability.reserve(leagal_actions.size());
                double max_logit = 0;
                for (const auto& leagal_action : leagal_actions) {
                    const auto logit = (double)(*policy)[(size_t)leagal_action.action];
                    probability.emplace_back(logit);
                    if (logit > max_logit)
                        max_logit = logit;
                }

                // オーバーフローを防止するため最大値で引く
                for (int i = 0; i < probability.size(); i++) {
                    auto& x = probability[i];
                    x = std::exp(x - max_logit);
                }
                // discrete_distributionを使用するためNormalizeは不要

                // サンプリング
                std::discrete_distribution<size_t> dist(probability.begin(), probability.end());
                const auto selected_index = dist(mt);

                // 行動
                const auto& action = leagal_actions[selected_index];
                if (action.action != Action::NO_ACTION) {
                    const auto& reply = action.reply;
                    game.reply(player_id, reply.msg, reply.arg);
                }

                policy++;
            }
        }

        // 遷移
        for (int game_id = 0; game_id < n_games; game_id++) {
            auto& game = games[game_id];
            game.next();

            // ゲーム終了時、牌譜出力
            if (game.status() == Game::Status::JIEJI) {
                std::lock_guard<std::mutex> lock(paipu_mutex);
                paipu.emplace_back(std::move(game.paipu()));
            }
        }
    }

    checkCudaErrors(cudaFreeHost(features_batch));
    checkCudaErrors(cudaFreeHost(policy_batch));
}

void output_paipu(const std::filesystem::path& path, const int id, const std::vector<Game::Paipu>& paipu_tmp) {
    auto paipu_path = path;
    paipu_path /= std::to_string(id) + "." + get_current_datetime() + ".paipu";
    std::ofstream ofs(paipu_path.string(), std::ios::binary);
    if (ofs) {
        for (const auto& paipu_ : paipu_tmp) {
            ofs << paipu_;
        }
    }
}

int main(int argc, char** argv)
{
    std::string basedir, modelfile;
    int n_games;
    int n_threads;
    int id;
    int interval;

    cxxopts::Options options("collect_rollouts", "Collect rollouts");
    options.positional_help("basedir modelfile");
    try {
        options.add_options()
        ("basedir", "Base directory", cxxopts::value<std::string>(basedir))
        ("modelfile", "Model file name", cxxopts::value<std::string>(modelfile))
        ("games", "Number of parallel games", cxxopts::value<int>(n_games)->default_value("64"))
        ("threads", "Number of threads", cxxopts::value<int>(n_threads)->default_value("2"))
        ("id", "Process id", cxxopts::value<int>(id)->default_value("0"))
        ("interval", "Output interval", cxxopts::value<int>(interval)->default_value("1000"))
        ("h,help", "Print help")
        ;
        options.parse_positional({ "basedir", "modelfile" });
        const auto result = options.parse(argc, argv);
        if (result.count("help")) {
            std::cout << options.help({}) << std::endl;
            return 0;
        }
    }
    catch (cxxopts::OptionException& e) {
        std::cout << options.usage() << std::endl;
        std::cerr << e.what() << std::endl;
        return 0;
    }

    const int max_batch_size = n_games * 3;

    while (true) {
        // basedir配下のディレクトリ名が数値のディレクトリ一覧を取得し、
        // 最もディレクトリ名の数値が大きいディレクトリにstartファイルが作成されたら処理を開始する
        // stopファイルが作成されたら停止する
        std::filesystem::path path;
        int max_dir_num = -1;
        try {
            for (const auto& entry : std::filesystem::directory_iterator(basedir)) {
                if (std::filesystem::is_directory(entry)) {
                    try {
                        int dir_num = std::stoi(entry.path().filename().c_str());
                        if (dir_num > max_dir_num) {
                            max_dir_num = dir_num;
                            path = entry.path();
                        }
                    }
                    catch (const std::exception& e) {
                    }
                }
            }
        }
        catch (std::filesystem::filesystem_error& e) {
            std::cerr << e.what() << std::endl;
        }
        if (max_dir_num < 0) {
            path = basedir;
            path /= std::to_string(0);
        }

        // stopファイルがある場合は、次のディレクトリ
        {
            auto stopfile_path = path;
            stopfile_path /= "stop";
            if (std::filesystem::exists(stopfile_path)) {
                max_dir_num++;
                path = basedir;
                path /= std::to_string(max_dir_num);
            }
        }

        // startファイルが作成されるまで待機
        auto startfile_path = path;
        startfile_path /= "start";
        while (!std::filesystem::exists(startfile_path)) {
            // ファイルが存在しない場合、待機
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // モデル読み込み
        auto modelfile_path = path;
        modelfile_path /= modelfile;
        PolicyInference inference{ modelfile_path.string().c_str() , max_batch_size};

        // 処理開始
        std::vector<std::thread> workers;
        workers.reserve(n_threads);
        for (int i = 0; i < n_threads; i++)
            workers.emplace_back(collect_rollouts, std::ref(inference), n_games);

        // stopファイルが作成されるまで待機
        auto stopfile_path = path;
        stopfile_path /= "stop";
        while (!std::filesystem::exists(stopfile_path)) {
            // ファイルが存在しない場合、待機
            std::this_thread::sleep_for(std::chrono::seconds(1));

            // 牌譜出力
            if (paipu.size() >= interval) {
                paipu_mutex.lock();
                auto paipu_tmp = std::move(paipu);
                paipu_mutex.unlock();
                output_paipu(path, id, paipu_tmp);
            }
        }

        // 停止
        stop = true;
        for (auto& worker : workers) {
            worker.join();
        }
        stop = false;

        // 牌譜出力
        output_paipu(path, id, paipu);
    }

    return 0;
}
