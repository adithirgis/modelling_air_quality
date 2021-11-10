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
  filter(!is.nan(BAM)) %>%
  select(everything(), -date, -BC, -PA_CF_1)

file_shared$hour <- as.factor(file_shared$hour)


# h2o.shutdown()
h2o.no_progress()
h2o.init(max_mem_size = "5g")

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
                        max_runtime_secs = 90*60,
                        seed = 108)


# save and load the model
# model_path <- h2o.saveModel(object = model, path = getwd(), force = TRUE)
# print(model_path)

# saved_model <- h2o.loadModel(model_path)

# my_local_model <- h2o.download_model(model, path = "/Users/UserName/Desktop")

# uploaded_model <- h2o.upload_model(my_local_model)
