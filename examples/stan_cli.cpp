#include <walnuts/adaptive_walnuts.hpp>
#include <walnuts/nuts.hpp>
#include <walnuts/walnuts.hpp>

#include "load_stan.hpp"

#include <CLI/CLI.hpp>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

static void summarize(const std::vector<std::string> names,
                      const Eigen::MatrixXd& draws) {
  auto N = draws.cols();
  auto D = draws.rows();
  for (auto d = 0; d < D; ++d) {
    if (d > 3 && d < D - 3) {
      if (d == 4) {
        std::cout << "... elided " << (D - 6) << " rows ..." << std::endl;
      }
      continue;
    }
    auto mean = draws.row(d).mean();
    auto var = (draws.row(d).array() - mean).square().sum() / (N - 1);
    auto stddev = std::sqrt(var);
    std::cout << names[static_cast<std::size_t>(d)] << ": mean = " << mean
              << ", stddev = " << stddev << "\n";
  }
}

static void write_draws(const std::string& filename,
                        const std::vector<std::string>& names,
                        const Eigen::MatrixXd& draws) {
  if (filename.empty()) {
    return;
  }
  std::ofstream out(filename);
  if (!out) {
    std::cerr << "Failed to open output file: " << filename << std::endl;
    return;
  }

  for (size_t i = 0; i < names.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << names[i];
  }
  out << "\n";

  auto EigenCommaFormat =
      Eigen::IOFormat(12, Eigen::DontAlignCols, ",", "\n", "", "", "", "");

  out << draws.transpose().format(EigenCommaFormat);
  out.close();
}

template <typename RNG>
Eigen::MatrixXd run_walnuts(DynamicStanModel& model, RNG& rng,
                            const Eigen::VectorXd& theta_init,
                            std::size_t num_warmup, std::size_t num_draws,
                            std::size_t refresh, bool show_diagnostics,
                            double init_count, double mass_iteration_offset,
                            double additive_smoothing,
                            double metric_init_exponent, double metric_floor,
                            double step_size_init,
                            double accept_rate_target, double learn_rate,
                            double beta1, double beta2, double epsilon,
                            double max_error, std::size_t max_nuts_depth,
                            std::size_t max_step_depth,
                            std::size_t min_micro_steps, double target_depth,
                            bool use_windowed_adapt, std::size_t initial_window,
                            std::size_t terminal_window, std::size_t base_window,
                            bool use_dual_averaging,
                            const std::string& output_file,
                            const std::string& warmup_diag_file) {
  using Clock = std::chrono::high_resolution_clock;
  auto elapsed_seconds = [](auto t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
  };
  double logp_time = 0.0;
  std::size_t logp_count = 0;
  auto global_start = Clock::now();

  auto end_timing = [&]() {
    auto global_total_time = elapsed_seconds(global_start);
    std::cout << "    total time: " << global_total_time << "s" << std::endl;
    std::cout << "logp_grad time: " << logp_time << "s" << std::endl;
    std::cout << "logp_grad fraction: " << logp_time / global_total_time
              << std::endl;
    std::cout << "        logp_grad calls: " << logp_count << std::endl;
    std::cout << "        time per call: " << logp_time / logp_count << "s"
              << std::endl;
    std::cout << std::endl;
  };

  Eigen::VectorXd mass_init = Eigen::VectorXd::Ones(theta_init.size());
  nuts::MassAdaptConfig mass_cfg(mass_init, init_count, mass_iteration_offset,
                                 additive_smoothing, metric_init_exponent,
                                 metric_floor);
  if (metric_init_exponent < 1e-10) {
    std::cout << "Using unit (identity) mass matrix initialization (Stan-style)"
              << std::endl;
  } else {
    std::cout << "Using gradient-based mass matrix initialization with exponent "
              << metric_init_exponent << " (0.5 = nutpie-style)" << std::endl;
  }

  nuts::WalnutsConfig walnuts_cfg(max_error, max_nuts_depth, max_step_depth,
                                  min_micro_steps);

  // Configure windowed adaptation
  nuts::WindowedAdaptConfig window_cfg;
  window_cfg.enabled = use_windowed_adapt;
  window_cfg.initial_window = initial_window;
  window_cfg.terminal_window = terminal_window;
  window_cfg.base_window = base_window;
  if (use_windowed_adapt) {
    std::cout << "Using windowed adaptation (Stan-style): initial="
              << initial_window << ", terminal=" << terminal_window
              << ", base=" << base_window << std::endl;
  } else {
    std::cout << "Using continuous adaptation (Nutpie-style)" << std::endl;
  }

  std::cout << "Step optimizer: "
            << (use_dual_averaging ? "Dual Averaging" : "Adam") << std::endl;
  std::cout << "Running Adaptive WALNUTS"
            << ";  D = " << theta_init.size() << "; W = " << num_warmup
            << ";  N = " << num_draws << "; step_size_init = " << step_size_init
            << "; max_nuts_depth = " << max_nuts_depth
            << "; max_error = " << max_error << std::endl;

  auto logp = [&](auto&&... args) {
    auto start = Clock::now();
    model.logp_grad(args...);
    logp_time += elapsed_seconds(start);
    ++logp_count;
  };

  // Create warmup diagnostics file if requested
  std::ofstream warmup_diag_out;
  bool write_warmup_diag = !warmup_diag_file.empty();
  if (write_warmup_diag) {
    warmup_diag_out.open(warmup_diag_file);
    if (warmup_diag_out) {
      warmup_diag_out
          << "iteration,step_size,metric_cond_num,metric_ess,tree_depth,"
             "max_energy_error,divergent_count,total_micro_steps,window_boundary"
          << std::endl;
    } else {
      std::cerr << "Warning: Failed to open warmup diagnostics file: "
                << warmup_diag_file << std::endl;
      write_warmup_diag = false;
    }
  }

  // Create adaptive sampler with appropriate step optimizer
  std::unique_ptr<nuts::AdaptiveWalnuts<decltype(logp), double, RNG>> walnuts_ptr;
  if (use_dual_averaging) {
    walnuts_ptr = std::make_unique<nuts::AdaptiveWalnuts<decltype(logp), double, RNG>>(
        rng, logp, theta_init, mass_cfg, step_size_init, accept_rate_target,
        walnuts_cfg, target_depth, window_cfg, num_warmup);
  } else {
    nuts::AdamConfig step_cfg(step_size_init, accept_rate_target, learn_rate,
                              beta1, beta2, epsilon);
    walnuts_ptr = std::make_unique<nuts::AdaptiveWalnuts<decltype(logp), double, RNG>>(
        rng, logp, theta_init, mass_cfg, step_cfg, walnuts_cfg, target_depth,
        window_cfg, num_warmup);
  }
  auto& walnuts = *walnuts_ptr;
  Eigen::VectorXd prev_theta;
  if (show_diagnostics && refresh > 0) {
    prev_theta = theta_init;
  }
  std::size_t total_divergences = 0;
  for (std::size_t w = 0; w < num_warmup; ++w) {
    nuts::WarmupIterationDiagnostics<double> iter_diag;
    Eigen::VectorXd new_theta = walnuts(iter_diag);
    total_divergences += iter_diag.divergent_count;

    // Write per-iteration diagnostics to file
    if (write_warmup_diag) {
      warmup_diag_out << iter_diag.iteration << "," << iter_diag.step_size << ","
                      << iter_diag.metric_condition_number << ","
                      << iter_diag.metric_effective_n << ","
                      << iter_diag.tree_depth << ","
                      << iter_diag.max_energy_error << ","
                      << iter_diag.divergent_count << ","
                      << iter_diag.total_micro_steps << ","
                      << (iter_diag.window_boundary ? 1 : 0) << std::endl;
    }

    if (refresh > 0 && (w + 1) % refresh == 0) {
      if (show_diagnostics) {
        double step_sz = walnuts.step_size();
        double dist_moved = (new_theta - prev_theta).norm();
        std::cout << "Warmup iteration: " << (w + 1) << " / " << num_warmup
                  << ", step_size=" << step_sz << ", dist_moved=" << dist_moved
                  << ", cond_num=" << iter_diag.metric_condition_number
                  << ", energy_err=" << iter_diag.max_energy_error;
        if (iter_diag.divergent_count > 0) {
          std::cout << ", DIVERGENT";
        }
        if (iter_diag.window_boundary) {
          std::cout << ", [WINDOW RESET]";
        }
        std::cout << std::endl;
        prev_theta = new_theta;
      } else {
        std::cout << "Warmup iteration: " << (w + 1) << " / " << num_warmup
                  << std::endl;
      }
    }
  }
  end_timing();

  if (write_warmup_diag) {
    warmup_diag_out.close();
    std::cout << "Warmup diagnostics written to: " << warmup_diag_file
              << std::endl;
  }
  if (total_divergences > 0) {
    std::cout << "WARNING: " << total_divergences
              << " divergent transitions during warmup" << std::endl;
  }

  // Check step size before creating sampler (to diagnose crashes)
  double final_step_size = walnuts.step_size();
  std::cout << "Final adapted step size: " << final_step_size << std::endl;
  std::cout << "Final metric condition number: "
            << walnuts.metric_condition_number() << std::endl;
  if (!std::isfinite(final_step_size) || final_step_size <= 0) {
    std::cerr << "ERROR: Step size adaptation failed (step_size="
              << final_step_size << ")" << std::endl;
    std::cerr << "This usually indicates the mass matrix initialization caused "
                 "problems."
              << std::endl;
    std::cerr << "Try running with --metric-init-exponent 0 (unit metric)."
              << std::endl;
    return Eigen::MatrixXd();  // Return empty matrix to signal failure
  }

  // N post-warmup draws
  auto sampler = walnuts.sampler();  // freeze tuning
  std::cout << "Adaptation completed." << std::endl;
  std::cout << "Macro step size = " << sampler.macro_step_size() << std::endl;

  // Write mass matrix diagonal to diagnostic file (too large for console)
  auto mass_diag = sampler.inverse_mass_matrix_diagonal();
  std::cout << "Mass matrix diagonal: " << mass_diag.size() << " elements";
  if (mass_diag.size() > 0) {
    std::cout << " (min=" << mass_diag.minCoeff()
              << ", max=" << mass_diag.maxCoeff()
              << ", mean=" << mass_diag.mean() << ")";
  }
  std::cout << std::endl;

  if (!output_file.empty()) {
    std::string diag_file = output_file;
    // Replace .csv with _diagnostics.csv
    auto pos = diag_file.rfind(".csv");
    if (pos != std::string::npos) {
      diag_file.replace(pos, 4, "_diagnostics.csv");
    } else {
      diag_file += "_diagnostics.csv";
    }
    std::ofstream diag_out(diag_file);
    if (diag_out) {
      diag_out << "mass_matrix_diagonal\n";
      for (Eigen::Index i = 0; i < mass_diag.size(); ++i) {
        diag_out << mass_diag(i) << "\n";
      }
      diag_out.close();
      std::cout << "Wrote mass matrix diagonal to: " << diag_file << std::endl;
    }
  }

  Eigen::MatrixXd draws(model.constrained_dimensions(), num_draws);

  logp_time = 0.0;
  logp_count = 0;
  global_start = Clock::now();

  for (std::size_t n = 0; n < num_draws; ++n) {
    model.constrain_draw(sampler(), draws.col(static_cast<Eigen::Index>(n)));
    if (refresh > 0 && (n + 1) % refresh == 0) {
      std::cout << "Sampling iteration: " << (n + 1) << " / " << num_draws << std::endl;
    }
  }

  end_timing();

  return draws;
}

template <typename RNG>
Eigen::VectorXd initialize(DynamicStanModel& model, RNG& rng, double init_range,
                           std::size_t max_tries = 100) {
  std::size_t D = model.unconstrained_dimensions();
  std::uniform_real_distribution<double> initial(-init_range, init_range);
  Eigen::VectorXd theta_init(D);

  Eigen::VectorXd grad(D);
  double logp = 0.0;

  for (std::size_t _ = 0; _ < max_tries; ++_) {
    for (std::size_t i = 0; i < D; ++i) {
      theta_init(static_cast<Eigen::Index>(i)) = initial(rng);
    }

    model.logp_grad(theta_init, logp, grad);
    if (std::isfinite(logp) && grad.allFinite()) {
      // if the log density and gradient are finite, we can use this
      // as the initial point
      std::cout << "Initialized at [" << theta_init.transpose() << "]"
                << std::endl;
      return theta_init;
    }
  }

  throw std::runtime_error("Failed to initialize the model after " +
                           std::to_string(max_tries) + " tries.");
}

int main(int argc, char** argv) {
  auto clock_count =
      std::chrono::system_clock::now().time_since_epoch().count();
  auto clock_seed = static_cast<unsigned int>(clock_count);
  srand(clock_seed);
  auto seed = static_cast<unsigned int>(rand());
  std::size_t num_warmup = 128;
  std::size_t num_draws = 128;
  std::size_t refresh = 100;
  std::size_t max_nuts_depth = 10;
  std::size_t max_step_depth = 8;
  std::size_t min_micro_steps = 1;
  double max_error = 0.5;
  double init = 2.0;
  double mass_init_count = 1.1;
  double mass_iteration_offset = 1.1;
  double mass_additive_smoothing = 1e-5;
  double metric_init_exponent = 0.5;  // 0 = unit, 0.5 = nutpie, 1 = full grad
  double metric_floor = 1e-8;
  bool show_diagnostics = false;
  double step_size_init = 1.0;
  double accept_rate_target = 0.8;
  double step_learn_rate = 0.2;
  double step_beta1 = 0.3;
  double step_beta2 = 0.99;
  double step_epsilon = 1e-4;
  double target_depth = 3.5;
  bool use_windowed_adapt = false;
  std::size_t initial_window = 75;
  std::size_t terminal_window = 50;
  std::size_t base_window = 25;
  bool use_dual_averaging = false;

  std::string lib;
  std::string data;
  std::string output_file;
  std::string init_json;
  std::string warmup_diag_file;

  // parse from command line with CLI11
  {
    CLI::App app{"Run WALNUTs on a Stan model"};

    app.add_option("--seed", seed, "Random seed (default randomize with clock)")
        ->default_val(seed);

    app.add_option("--warmup", num_warmup, "Number of warmup iterations")
        ->default_val(num_warmup)
        ->check(CLI::NonNegativeNumber);

    app.add_option("--samples", num_draws, "Number of samples to draw")
        ->default_val(num_draws)
        ->check(CLI::PositiveNumber);

    app.add_option("--refresh", refresh, "Number of iterations between progress reports")
        ->default_val(refresh)
        ->check(CLI::PositiveNumber);

    app.add_option("--max-depth", max_nuts_depth,
                   "Maximum depth for NUTS trajectory doublings")
        ->default_val(max_nuts_depth)
        ->check(CLI::PositiveNumber);

    app.add_option("--max-step-depth", max_step_depth,
                   "Maximum depth for the step size adaptation")
        ->default_val(max_step_depth)
        ->check(CLI::PositiveNumber);

    app.add_option("--min-micro-steps", min_micro_steps,
                   "Minimum micro steps per macro step")
        ->default_val(min_micro_steps)
        ->check(CLI::PositiveNumber);

    app.add_option("--max-error", max_error,
                   "Maximum error allowed in joint densities")
        ->default_val(max_error)
        ->check(CLI::PositiveNumber);

    app.add_option("--init", init,
                   "Range [-init,init] for uniform parameter initial values")
        ->default_val(init)
        ->check(CLI::NonNegativeNumber);

    app.add_option("--mass-init-count", mass_init_count,
                   "Initial count for the mass matrix adaptation")
        ->default_val(mass_init_count)
        ->check(CLI::Range(1.0, (std::numeric_limits<double>::max)()));

    app.add_option("--mass-iteration-offset", mass_iteration_offset,
                   "Offset for the mass matrix adaptation iterations")
        ->default_val(mass_iteration_offset)
        ->check(CLI::Range(1.0, (std::numeric_limits<double>::max)()));

    app.add_option("--mass-additive-smoothing", mass_additive_smoothing,
                   "Additive smoothing for the mass matrix adaptation")
        ->default_val(mass_additive_smoothing)
        ->check(CLI::PositiveNumber);

    app.add_option("--metric-init-exponent", metric_init_exponent,
                   "Exponent for gradient-based metric initialization. "
                   "0 = unit metric (Stan-style), 0.5 = geometric mean (nutpie), "
                   "1 = full gradient-based")
        ->default_val(metric_init_exponent)
        ->check(CLI::Range(0.0, 1.0));

    app.add_option("--metric-floor", metric_floor,
                   "Minimum value for metric diagonal entries")
        ->default_val(metric_floor)
        ->check(CLI::PositiveNumber);

    app.add_flag("--diagnostics", show_diagnostics,
                 "Show detailed diagnostics at each refresh interval");

    app.add_flag("--windowed-adapt", use_windowed_adapt,
                 "Use windowed adaptation (Stan-style) instead of continuous "
                 "(Nutpie-style). Resets mass estimator at window boundaries.");

    app.add_option("--initial-window", initial_window,
                   "Size of initial adaptation window (only with --windowed-adapt)")
        ->default_val(initial_window)
        ->check(CLI::PositiveNumber);

    app.add_option("--terminal-window", terminal_window,
                   "Size of terminal adaptation window (only with --windowed-adapt)")
        ->default_val(terminal_window)
        ->check(CLI::PositiveNumber);

    app.add_option("--base-window", base_window,
                   "Base size for doubling windows (only with --windowed-adapt)")
        ->default_val(base_window)
        ->check(CLI::PositiveNumber);

    app.add_flag("--dual-averaging", use_dual_averaging,
                 "Use dual averaging for step size adaptation instead of Adam");

    app.add_option("--step-size-init", step_size_init,
                   "Initial step size for the step size adaptation")
        ->default_val(step_size_init)
        ->check(CLI::PositiveNumber);

    app.add_option("--step-accept-rate-target", accept_rate_target,
                   "Target acceptance rate for the step size adaptation")
        ->default_val(accept_rate_target)
        ->check(CLI::Range((std::numeric_limits<double>::min)(), 1.0));

    app.add_option("--step-learning-rate", step_learn_rate,
                   "Learning rates for step adaptation")
        ->default_val(step_learn_rate)
        ->check(CLI::PositiveNumber);

    app.add_option("--step-beta1", step_beta1,
                   "Decay rate of gradient moving average for step adaptation")
        ->default_val(step_beta1)
        ->check(CLI::Range((std::numeric_limits<double>::min)(), 1.0));

    app.add_option(
           "--step-beta2", step_beta2,
           "Decay rate of squared gradient moving average for step adaptation")
        ->default_val(step_beta2)
        ->check(CLI::Range((std::numeric_limits<double>::min)(), 1.0));

    app.add_option("--step-epsilon", step_epsilon,
                   "Update stabilization term for step size adaptation")
        ->default_val(step_epsilon)
        ->check(CLI::PositiveNumber);

    app.add_option("--target-depth", target_depth,
                   "Target number of trajectory doublings in NUTS.")
        ->default_val(target_depth)
        ->check(CLI::PositiveNumber);

    app.add_option("model", lib,
                   "Path to the Stan model library (.so from CmdStan{,Py,R})")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("data", data,
                   "Path to the Stan model data (.json, optional)")
        ->check(CLI::ExistingFile);

    app.add_option("--output", output_file, "Output file for the draws")
        ->check(CLI::NonexistentPath);

    app.add_option("--init-json", init_json,
                   "Path to the Stan model initialization (.json)")
        ->check(CLI::ExistingFile);

    app.add_option("--warmup-diag", warmup_diag_file,
                   "Output CSV file for per-iteration warmup diagnostics");

    CLI11_PARSE(app, argc, argv);
  }

  DynamicStanModel model(lib.c_str(), data.c_str(), seed);

  std::mt19937 rng(seed);

  Eigen::VectorXd theta_init(model.unconstrained_dimensions());
  if (!init_json.empty()) {
    std::ifstream ifs(init_json);
    if (!ifs.is_open()) {
      throw std::runtime_error("Could not open init JSON file: " + init_json);
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    std::string json_content = buffer.str();

    std::cout << "Read " << json_content.size() << " bytes from init JSON file." << std::endl;

    model.param_unconstrain(json_content, theta_init);
    std::cout << "Initialized parameters from JSON file." << std::endl;

    // Diagnostic: print first few unconstrained parameters
    std::cout << "Unconstrained dimensions: " << theta_init.size() << std::endl;
    std::cout << "First 10 unconstrained values: ";
    for (Eigen::Index i = 0; i < std::min(static_cast<Eigen::Index>(10), theta_init.size()); ++i) {
      std::cout << theta_init(i) << " ";
    }
    std::cout << std::endl;

    // Check for NaN/Inf in initial values
    bool has_nan = !theta_init.allFinite();
    if (has_nan) {
      std::cerr << "WARNING: Initial parameters contain NaN or Inf values!" << std::endl;
    }

    // Compute initial log probability to verify model accepts these inits
    double init_logp = 0.0;
    Eigen::VectorXd init_grad(theta_init.size());
    model.logp_grad(theta_init, init_logp, init_grad);
    std::cout << "Initial log probability: " << init_logp << std::endl;

    if (!std::isfinite(init_logp)) {
      std::cerr << "WARNING: Initial log probability is not finite!" << std::endl;
      std::cerr << "This means the model rejects these initial values." << std::endl;
    }

    bool grad_finite = init_grad.allFinite();
    if (!grad_finite) {
      std::cerr << "WARNING: Initial gradient contains NaN or Inf values!" << std::endl;
      // Find which parameters have bad gradients
      int bad_count = 0;
      for (Eigen::Index i = 0; i < init_grad.size() && bad_count < 10; ++i) {
        if (!std::isfinite(init_grad(i))) {
          std::cerr << "  Parameter " << i << " has gradient: " << init_grad(i) << std::endl;
          bad_count++;
        }
      }
    }
  } else {
    theta_init = initialize(model, rng, init);
  }

  Eigen::MatrixXd draws = run_walnuts(
      model, rng, theta_init, num_warmup, num_draws, refresh, show_diagnostics,
      mass_init_count, mass_iteration_offset, mass_additive_smoothing,
      metric_init_exponent, metric_floor, step_size_init, accept_rate_target,
      step_learn_rate, step_beta1, step_beta2, step_epsilon, max_error,
      max_nuts_depth, max_step_depth, min_micro_steps, target_depth,
      use_windowed_adapt, initial_window, terminal_window, base_window,
      use_dual_averaging, output_file, warmup_diag_file);

  auto names = model.param_names();
  summarize(names, draws);
  write_draws(output_file, names, draws);

  return 0;
}
