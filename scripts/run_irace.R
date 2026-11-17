# Project-specific bridge to the official irace targetRunner API.
# Interface references: irace examples/slurm/run_irace.R and jMetal Evolver irace tutorial.
# No foreign solver or configurator implementation is copied into this project.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) stop("Usage: Rscript run_irace.R campaign.json [recover]")
suppressPackageStartupMessages(library(irace))
suppressPackageStartupMessages(library(jsonlite))
settings <- fromJSON(args[1L], simplifyVector = FALSE)
dir.create(settings$output, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(PYTHONPATH = settings$src, OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
runner <- function(experiment, scenario) {
  key <- paste(experiment$id_configuration, experiment$id_instance, experiment$seed, sep = "_")
  request_file <- file.path(settings$output, paste0("request_", key, ".json"))
  request <- list(configuration = as.list(experiment$configuration),
                  seed = as.integer(experiment$seed), case = experiment$instance,
                  seal = settings$seal,
                  retry_incomplete = length(args) >= 2L && args[2L] == "recover",
                  run_dir = file.path(settings$output, "runs", key))
  write_json(request, request_file, auto_unbox = TRUE, pretty = TRUE)
  stdout <- file.path(settings$output, paste0("stdout_", key, ".log"))
  stderr <- file.path(settings$output, paste0("stderr_", key, ".log"))
  status <- system2(settings$python, c(shQuote(settings$target), shQuote(request_file)),
                    stdout = stdout, stderr = stderr)
  if (status != 0) stop(paste("Target failed:", key, "see", stderr))
  value <- suppressWarnings(as.numeric(readLines(stdout, warn = FALSE)))
  if (length(value) != 1L || !is.finite(value)) stop("Target did not return one finite cost")
  list(cost = value)
}
scenario <- list(parameters = readParameters(settings$parameters), targetRunner = runner,
                 trainInstancesFile = settings$instances, trainInstancesDir = "",
                 configurationsFile = settings$defaults,
                 maxExperiments = as.integer(settings$max_experiments),
                 seed = as.integer(settings$seed), parallel = 1L,
                 sampleInstances = 0L, blockSize = as.integer(settings$block_size),
                 firstTest = as.integer(settings$first_test), eachTest = 1L,
                 capping = FALSE, elitist = TRUE, postselection = FALSE,
                 logFile = file.path(settings$output, "irace.Rdata"),
                 execDir = settings$output)
if (!is.null(settings$nb_configurations)) scenario$nbConfigurations <- as.integer(settings$nb_configurations)
if (!is.null(settings$nb_iterations)) scenario$nbIterations <- as.integer(settings$nb_iterations)
if (!is.null(settings$min_survival)) scenario$minNbSurvival <- as.integer(settings$min_survival)
if (length(args) >= 2L && args[2L] == "recover") {
  snapshot <- paste0(scenario$logFile, ".recovery_", format(Sys.time(), "%Y%m%d_%H%M%OS6"))
  if (!file.copy(scenario$logFile, snapshot, overwrite = FALSE)) stop("Cannot preserve recovery checkpoint")
  scenario$recoveryFile <- snapshot
} else if (file.exists(scenario$logFile)) stop("Existing irace log: use recover or a new output")
write_json(list(R = R.version.string, irace = as.character(packageVersion("irace"))),
           file.path(settings$output, "runtime.json"), auto_unbox = TRUE)
result <- irace_main(scenario = scenario)
saveRDS(result, file.path(settings$output, "result.rds"))
# irace_main returns the ranked elite data frame (the full state is in irace.Rdata).
write.csv(result, file.path(settings$output, "elites.csv"), row.names = FALSE)
