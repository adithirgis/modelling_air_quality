library(xgboost)
library(readxl)
library(tidyverse)
library(tidymodels)
library(h2o)
library(here)
library(data.table)

set.seed(108)

file_shared <- read_csv(here("data", "PurpleAir_BAM_Data.csv")) %>%
  select("date" = `Date and time (Before time stamp)`, 
         "BAM" = `BAM-PM2.5`, "PA_CF_1" = `PA-PM2.5 (CF=1)`, "PA_CF_ATM" = `PA-PM2.5 (CF=ATM)`, 
         "PA_RH" = `PA-RH`, "PA_Temp" = `PA-Temperature`, BC) %>%
  mutate(date = as.POSIXct(date, format = "%d-%m-%Y %H:%M", tz = "Asia/Kolkata"),
         hour = format(date, "%H")) %>% 
  mutate_if(is.character, as.numeric) %>% 
  na.omit() %>%
  select(BAM, everything(), -date, -BC, -PA_CF_1)


file_shared$hour <- as.factor(file_shared$hour)


# h2o.shutdown()
h2o.no_progress()
h2o.init(max_mem_size = "6g")

file_shared <- as.h2o(file_shared)

splits <- h2o.splitFrame(
  data = file_shared,
  ratios = c(0.7, 0),   
  destination_frames = c("train_hex", "valid_hex", "test_hex"), seed = 108
)
train <- splits[[1]]
valid <- splits[[2]]
test  <- splits[[3]]

response <- "BAM"
features <- setdiff(names(train), c(response))
h2o.describe(file_shared)

search_criteria <- list(strategy = "RandomDiscrete", 
                        stopping_metric = "mse",
                        stopping_tolerance = 1e-4,
                        stopping_rounds = 10,
                        max_runtime_secs = 60*60,
                        seed = 108)



hyper_grid <- list(
  activation = c("Rectifier", "Tanh", "RectifierWithDropout", "TanhWithDropout"), 
  hidden = list(c(5, 5, 5, 5, 5), c(30, 30, 30, 30), c(50, 50, 50, 50)),
  epochs = c(50, 100, 200, 300, 400, 500),
  l1 = c(0, 0.00001), 
  l2 = c(0, 0.00001),
  rate = c(0, 0.1, 0.005, 0.001),
  rate_annealing = c(1e-7, 1e-6),
  rho = c(0.9, 0.95, 0.99, 0.999),
  epsilon = c(1e-10, 1e-8, 1e-6),
  momentum_start = c(0.5),
  momentum_stable = c(0.99, 0.5),
  input_dropout_ratio = c(0.1, 0.2),
  max_w2 = c(1, 10, 100, 1000, 3.4028235e+38)
)

start <- Sys.time()
dl_grid <- h2o.grid(algorithm = "deeplearning", 
                    x = features,
                    y = response,
                    grid_id = "dl_grid",
                    training_frame = train,
                    nfolds = 10,                           
                    hyper_params = hyper_grid,
                    search_criteria = search_criteria,
                    seed = 108
)
end <- Sys.time()

grid_perf <- h2o.getGrid(
  grid_id = "dl_grid", 
  sort_by = "mse", 
  decreasing = FALSE
)
print(grid_perf)

best_model_id <- grid_perf@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)

model_path <- h2o.saveModel(object = best_model, path = getwd(), force = TRUE)
print(model_path)
saved_model <- h2o.loadModel(model_path)

h2o.scoreHistory(best_model)
plot(best_model, 
     timestep = "epochs", 
     metric = "rmse")

cv_models <- sapply(best_model@model$cross_validation_models, 
                    function(i) h2o.getModel(i$name))

plot(cv_models[[1]], 
     timestep = "epochs", 
     metric = "rmse")

best_model_perf <- h2o.performance(model = best_model, newdata = test)

h2o.mse(best_model_perf) %>% sqrt()

file_shared$nn <- predict(best_model, file_shared)
file_shared <- as.data.frame(file_shared)
ggplot(file_shared, aes(BAM, h2o_nn)) + geom_point() + geom_smooth(method = "lm")
summary(lm(BAM ~ h2o_nn, data = file_shared))
mean(abs((file_shared$BAM - file_shared$h2o_nn) / file_shared$BAM), na.rm = TRUE) * 100
write.csv(file_shared, "NN.csv")


