#include "../../cmajiang/src_cpp/game.h"
#include "../../cmajiang/src_cpp/random.h"
#include "../../cmajiang/src_cpp/xiangting.h"

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

#include <zlib.h>

// PPO Parameter
float gamma = 0.99;
float gae_lambda = 0.95;

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

typedef float hupai_t[54];
typedef float hule_player_t[5];
typedef float tajia_tingpai_t[3][34];
struct StepData {
    public_features_t public_features;
    private_features_t private_features;
    Action action;
    float value;
    policy_t log_probs;
    float advantage;
    hupai_t hupai; // 役
    hule_player_t hule_player; // 和了プレイヤー
    tajia_tingpai_t tajia_tingpai; // 他家の待ち牌
    float fenpei[4]; // 得点
};

typedef std::vector<StepData> RolloutData;
typedef std::vector<RolloutData> RolloutBuffer;
RolloutBuffer rollout_buffer;
std::mutex rollout_buffer_mutex;

std::ostream& operator<<(std::ostream& os, const StepData& step_data) {
    os.write((const char*)step_data.public_features, sizeof(public_features_t));
    os.write((const char*)step_data.private_features, sizeof(private_features_t));
    const int64_t action = (int64_t)step_data.action;
    os.write((const char*)&action, sizeof(action));
    os.write((const char*)&step_data.value, sizeof(float));
    os.write((const char*)step_data.log_probs, sizeof(policy_t));
    os.write((const char*)&step_data.advantage, sizeof(float));
    os.write((const char*)step_data.hupai, sizeof(hupai_t));
    os.write((const char*)step_data.hule_player, sizeof(hule_player_t));
    os.write((const char*)step_data.tajia_tingpai, sizeof(tajia_tingpai_t));
    os.write((const char*)step_data.fenpei, sizeof(StepData::fenpei));
    return os;
}

std::ostream& operator<<(std::ostream& os, const RolloutData& rollout_data) {
    for (const auto& step_data : rollout_data)
        os << step_data;
    return os;
}

std::ostream& operator<<(std::ostream& os, const RolloutBuffer& rollout_buffer) {
    for (const auto& rollout_data : rollout_buffer)
        os << rollout_data;
    return os;
}

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

    void forward(const int batch_size, public_features_t* features1_batch, private_features_t* features2_batch, policy_t* policy_batch, float* value_batch) {
        std::lock_guard<std::mutex> lock(mutex);
        Inference::forward(batch_size, (float*)features1_batch, (float*)features2_batch, (float*)policy_batch, value_batch);
    }

    int get_max_batch_size() const { return max_batch_size; }

private:
    std::mutex mutex;
};

void compute_advantage(Game& game, std::array<RolloutData, 4>& rollout_data) {
    std::array<int, 4> fenpei{};
    hupai_t player_hupai[4]{};
    hule_player_t hule_player{};

    if (game.status() == Game::Status::HULE) {
        while (true) {
            const auto& hule = game.defen_();
            for (int l = 0; l < 4; l++)
                fenpei[l] += hule.fenpei[l];
            const auto player_id = game.player_id()[hule.menfeng];
            auto& player_hupai_ = player_hupai[player_id];
            for (const auto& hupai : hule.hupai) {
                if (hupai.name >= Hupai::ZHUANGFENG && hupai.name < Hupai::MENFENG)
                    player_hupai_[0] = 1;
                else if (hupai.name >= Hupai::MENFENG && hupai.name < Hupai::FANPAI)
                    player_hupai_[1] = 1;
                else if (hupai.name >= Hupai::FANPAI && hupai.name < Hupai::BAOPAI)
                    player_hupai_[2 + (int)(hupai.name - Hupai::FANPAI)] = 1;
                else if (hupai.name == Hupai::BAOPAI) {
                    for (int n = 0; n < std::min(hupai.fanshu, 4); n++)
                        player_hupai_[3 + (int)(hupai.name - Hupai::BAOPAI) * 4 + n] = 1;
                }
                else if (hupai.name == Hupai::CHIBAOPAI) {
                    for (int n = 0; n < std::min(hupai.fanshu, 3); n++)
                        player_hupai_[7 + (int)(hupai.name - Hupai::BAOPAI) * 4 + n] = 1;
                }
                else if (hupai.name == Hupai::LIBAOPAI) {
                    for (int n = 0; n < std::min(hupai.fanshu, 4); n++)
                        player_hupai_[10 + (int)(hupai.name - Hupai::BAOPAI) * 4 + n] = 1;
                }
                else
                    player_hupai_[14 + (int)(hupai.name - Hupai::LIZHI)] = 1;
            }
            hule_player[hule.menfeng] = 1;

            if (game.hule_().size() == 0)
                break;
            game.next();
        }
    }
    else {
        // 流局
        hule_player[4] = 1;
    }

    for (int player_id = 0; player_id < 4; player_id++) {
        auto& player_rollout_data = rollout_data[player_id];
        const auto lunban = game.player_lunban(player_id);
        float last_gae_lam = 0;
        for (int step = player_rollout_data.size() - 1; step >= 0; step--) {
            const bool next_non_terminal = step != player_rollout_data.size() - 1;
            const auto reward = next_non_terminal ? 0 : fenpei[lunban] / 12000.0f;
            const auto next_value = next_non_terminal ? player_rollout_data[step + 1].value : 0;
            const auto delta = reward + gamma * next_value - player_rollout_data[step].value;
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam;
            player_rollout_data[step].advantage = last_gae_lam;

            // 役
            std::copy(player_hupai[player_id], player_hupai[player_id] + sizeof(hupai_t) / sizeof(float), player_rollout_data[step].hupai);
            // 和了プレイヤー
            std::fill_n(player_rollout_data[step].hule_player, sizeof(hule_player_t) / sizeof(float), 0);
            if (hule_player[4] == 1) {
                player_rollout_data[step].hule_player[4] = 1;
            }
            else {
                for (int l = 0; l < 4; l++) {
                    if (hule_player[l] == 1) {
                        player_rollout_data[step].hule_player[(lunban - l + 4) % 4] = 1;
                    }
                }
            }
            // 得点
            for (int l = 0; l < 4; l++) {
                player_rollout_data[step].fenpei[(lunban - l + 4) % 4] = fenpei[l] / 12000.0f;
            }
        }
    }
}

void collect_rollouts(PolicyInference& inference, const int n_games) {
    std::vector<Game> games(n_games);
    public_features_t* features1_batch;
    private_features_t* features2_batch;
    policy_t* policy_batch;
    float* value_batch;
    const auto max_batch_size = inference.get_max_batch_size();
    checkCudaErrors(cudaHostAlloc((void**)&features1_batch, sizeof(public_features_t) * max_batch_size, cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc((void**)&features2_batch, sizeof(private_features_t) * max_batch_size, cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc((void**)&policy_batch, sizeof(policy_t) * max_batch_size, cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc((void**)&value_batch, sizeof(float) * max_batch_size, cudaHostAllocPortable));

    struct GamePlayerIdTbl {
        int game_id;
        int player_id;
        std::vector<ActionReply> leagal_actions;
    };
    std::vector<GamePlayerIdTbl> game_player_id_tbl;

    std::vector<std::array<RolloutData, 4>> game_player_rollout_data(n_games);

    std::random_device seed_gen;
    std::mt19937_64 mt(seed_gen());

    const int n_xiangting = 3;

    while (!stop) {
        game_player_id_tbl.clear();

        public_features_t* features1 = features1_batch;
        private_features_t* features2 = features2_batch;
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
                std::fill_n((float*)features1, sizeof(public_features_t) / sizeof(float), 0);
                public_features(game, lunban, (channel_t*)features1);
                // player id
                fill_channel(&(*features1)[N_CHANNELS_PUBLIC + player_id], 1);
                std::fill_n((float*)features2, sizeof(private_features_t) / sizeof(float), 0);
                private_features(game, lunban, (channel_t*)features2);

                features1++;
                features2++;
            }

        }

        if (game_player_id_tbl.size() > 0) {
            // 推論(並列実行しているゲームの全プレイヤー分をまとめて推論)
            inference.forward((const int)game_player_id_tbl.size(), features1_batch, features2_batch, policy_batch, value_batch);

            // 推論結果のpolicyからサンプリングして行動
            features1 = features1_batch;
            features2 = features2_batch;
            policy_t* policy = policy_batch;
            float* value = value_batch;
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

                // データ格納
                auto& rollout_data = game_player_rollout_data[game_id][player_id];
                auto& step_data = rollout_data.emplace_back();
                std::copy((float*)features1, (float*)(features1 + 1), (float*)step_data.public_features);
                std::copy((float*)features2, (float*)(features2 + 1), (float*)step_data.private_features);
                step_data.action = action.action;
                step_data.value = *value;
                std::copy((float*)policy, (float*)(policy + 1), (float*)step_data.log_probs);

                // 他家の待ち牌
                std::fill_n((float*)step_data.tajia_tingpai, sizeof(tajia_tingpai_t) / sizeof(float), 0);
                const auto player_lunban = game.player_lunban(player_id);
                for (int l = 1; l < 4; l++) {
                    const auto lunban = (player_lunban + l) % 4;
                    // 待ち牌
                    for (const auto& p : tingpai(game.shoupai_(lunban))) {
                        const auto s = p[0];
                        const int suit = index_of(s);
                        const int n = p[1] == '0' ? 4 : p[1] - '1';
                        step_data.tajia_tingpai[l - 1][9 * suit + n] = 1;
                    }
                }

                features1++;
                features2++;
                policy++;
                value++;
            }
        }

        // 遷移
        for (int game_id = 0; game_id < n_games; game_id++) {
            auto& game = games[game_id];
            game.next();

            // 1局終了時
            if (game.status() == Game::Status::HULE || game.status() == Game::Status::PINGJU) {
                auto& game_rollout_data =  game_player_rollout_data[game_id];
                compute_advantage(game, game_rollout_data);
                std::lock_guard<std::mutex> lock(rollout_buffer_mutex);
                for (auto& rollout_data : game_rollout_data)
                    rollout_buffer.emplace_back(std::move(rollout_data));
            }
        }
    }

    checkCudaErrors(cudaFreeHost(features1_batch));
    checkCudaErrors(cudaFreeHost(features2_batch));
    checkCudaErrors(cudaFreeHost(policy_batch));
    checkCudaErrors(cudaFreeHost(value_batch));
}

void output_rollout_buffer(const std::filesystem::path& path, const int id, const RolloutBuffer& rollout_buffer) {
    auto rollout_buffer_path = path;
    rollout_buffer_path /= get_current_datetime() + "." + std::to_string(id) + ".dat";
    auto rollout_buffer_path_tmp = rollout_buffer_path;
    rollout_buffer_path_tmp += ".tmp";
    std::ofstream ofs(rollout_buffer_path_tmp, std::ios::binary);
    if (ofs) {
        // 出力
        std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
        ss << rollout_buffer;

        // ストリームのサイズを求める
        ss.seekg(0, std::ios::end);
        std::streamsize size = ss.tellg();
        ss.seekg(0, std::ios::beg);

        // バイナリデータを格納するためのベクタを初期化
        std::vector<unsigned char> data(size);

        // データを読み込む
        ss.read(reinterpret_cast<char*>(data.data()), size);

        // zlibを使用して圧縮   
        uLongf compressed_size = compressBound(data.size());
        std::vector<unsigned char> compressed_data(compressed_size);
        compress(compressed_data.data(), &compressed_size, data.data(), data.size());

        // 圧縮データをファイルに保存
        ofs.write(reinterpret_cast<const char*>(compressed_data.data()), compressed_size);
        ofs.close();

        // ファイル名変更
        std::filesystem::rename(rollout_buffer_path_tmp, rollout_buffer_path);
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
        {
            bool first = true;
            while (!std::filesystem::exists(startfile_path)) {
                if (first) {
                    std::cout << "Waiting " << startfile_path.string().c_str() << std::endl;
                    first = false;
                }
                // ファイルが存在しない場合、待機
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        // モデル読み込み
        auto modelfile_path = path;
        modelfile_path /= modelfile;
        std::cout << "Loading " << modelfile_path.string().c_str() << std::endl;
        PolicyInference inference{ modelfile_path.string().c_str() , max_batch_size};

        // 処理開始
        std::cout << "Collecting rollouts " << path.string().c_str() << std::endl;
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

            // 出力
            if (rollout_buffer.size() >= interval) {
                rollout_buffer_mutex.lock();
                auto rollout_buffer_tmp = std::move(rollout_buffer);
                rollout_buffer_mutex.unlock();
                output_rollout_buffer(path, id, rollout_buffer_tmp);
            }
        }

        // 停止
        stop = true;
        for (auto& worker : workers) {
            worker.join();
        }
        stop = false;

        // 出力
        output_rollout_buffer(path, id, rollout_buffer);
    }

    return 0;
}
