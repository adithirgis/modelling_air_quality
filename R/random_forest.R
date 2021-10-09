# https://uc-r.github.io/random_forests

library(readxl)
library(tidyverse)
library(rsample)      # data splitting 
library(randomForest) # basic implementation
library(ranger)       # a faster implementation of randomForest
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # an extremely fast java-based platform
library(here)



set.seed(007)

# file_shared <- read_excel(here("data", "PurpleAir_BAM_Data.xlsx"), sheet = 1) %>%
#   select("BAM" = `BAM-PM2.5`, "PA_CF_1" = `PA-PM2.5 (CF=1)`, "PA_CF_ATM" = `PA-PM2.5 (CF=ATM)`, 
#          "PA_RH" = `PA-RH`, "PA_Temp" = `PA-Temperature`, BC) %>%
#   mutate_if(is.character, as.numeric) %>%
#   na.omit()

file_shared <- read_csv(here("data", "PurpleAir_BAM_Data.csv")) %>%
  select("date" = `Date and time (Before time stamp)`, 
         "BAM" = `BAM-PM2.5`, "PA_CF_1" = `PA-PM2.5 (CF=1)`, "PA_CF_ATM" = `PA-PM2.5 (CF=ATM)`, 
         "PA_RH" = `PA-RH`, "PA_Temp" = `PA-Temperature`, BC) %>%
  mutate(date = as.POSIXct(date, format = "%d-%m-%Y %H:%M", tz = "Asia/Kolkata"),
         hour = format(date, "%H")) %>% 
  mutate_if(is.character, as.numeric) %>%
  na.omit() %>%
  select(everything(), -date, -BC, -PA_CF_1)


split_data_try <- initial_split(file_shared, prop = .75)
training_data <- training(split_data_try)
testing_data  <- testing(split_data_try)

# 1.  Given training data set
# 2.  Select number of trees to build (ntrees)
# 3.  for i = 1 to ntrees do
# 4.  |  Generate a bootstrap sample of the original data
# 5.  |  Grow a regression tree to the bootstrapped data
# 6.  |  for each split do
# 7.  |  | Select m variables at random from all p variables
# 8.  |  | Pick the best variable/split-point among the m
# 9.  |  | Split the node into two child nodes
# 10. |  end
# 11. | Use typical tree model stopping criteria to determine when a tree is complete (but do not prune)
# 12. end


# default RF model
model_calibration <- randomForest(
  formula = BAM ~ .,
  data    = file_shared
)
model_calibration
plot(model_calibration)
# number of trees with lowest MSE
which.min(model_calibration$mse)
# RMSE of this optimal random fores
sqrt(model_calibration$mse[which.min(model_calibration$mse)])



# For our data 
split_data_2 <- initial_split(file_shared, .75)
# training data
training_data_2 <- analysis(split_data_2)
# validation data
validation_data <- assessment(split_data_2)
params_with_testing_x <- validation_data[setdiff(names(validation_data), "BAM")]
params_with_testing_y <- validation_data$BAM

# New model with oob error considered and also including testing data
model_oob_comp <- randomForest(
  formula = BAM ~ .,
  data    = file_shared,
  xtest   = params_with_testing_x,
  ytest   = params_with_testing_y
)
model_oob_comp
plot(model_oob_comp)
# extract OOB & validation errors
oob <- sqrt(model_oob_comp$mse)
validation <- sqrt(model_oob_comp$test$mse)

# compare error rates
tibble::tibble(
  `Out of Bag Error` = oob,
  `Test error` = validation,
  ntrees = 1:model_oob_comp$ntree
) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  scale_y_continuous() +
  xlab("Number of trees")

# `ntree`: number of trees. We want enough trees to stabilize the error but using too many trees is unnecessarily inefficient, especially when using large data sets.
# `mtry`: the number of variables to randomly sample as candidates at each split. When mtry = p the model equates to bagging. When mtry = 1 the split variable is completely random, so all variables get a chance but can lead to overly biased results. A common suggestion is to start with 5 values evenly spaced across the range from 2 to p.
# `sampsize`: the number of samples to train on. The default value is 63.25% of the training set since this is the expected value of unique observations in the bootstrap sample. Lower sample sizes can reduce the training time but may introduce more bias than necessary. Increasing the sample size can increase performance but at the risk of overfitting because it introduces more variance. Typically, when tuning this parameter we stay near the 60-80% range.
# `nodesize`: minimum number of samples within the terminal nodes. Controls the complexity of the trees. Smaller node size allows for deeper, more complex trees and smaller node results in shallower trees. This is another bias-variance tradeoff where deeper trees introduce more variance (risk of overfitting) and shallower trees introduce more bias (risk of not fully capturing unique patters and relatonships in the data).
# `maxnodes`: maximum number of terminal nodes. Another way to control the complexity of the trees. More nodes equates to deeper, more complex trees and less nodes result in shallower trees.

features <- setdiff(names(file_shared), "BAM")

m2 <- tuneRF(
  x          = file_shared[features],
  y          = file_shared$BAM,
  ntreeTry   = 500,
  mtryStart  = 2,
  stepFactor = 1.5,
  improve    = 0.01,
  trace      = TRUE      # to not show real-time progress 
)

file_shared$pred_randomForest <- predict(model_calibration, file_shared)
ggplot(file_shared, aes(BAM, pred_randomForest)) + geom_point() + geom_smooth(method = "lm")
summary(lm(BAM ~ pred_randomForest, data = file_shared))
mean(abs((file_shared$BAM - file_shared$pred_randomForest) / file_shared$BAM)) * 100


# Use ranger for Random Forest to get faster results
# hyperparameter grid search
hyper_grid <- expand.grid(
  mtry       = seq(1, 4, by = 1),
  node_size  = seq(3, 9, by = 2),
  sampe_size = c(.55, .632, .70, 0.75, .80),
  OOB_RMSE   = 0
)

for(i in 1:nrow(hyper_grid)) {
  # train model
  model <- ranger(
    formula         = BAM ~ ., 
    data            = file_shared, 
    num.trees       = 500,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sampe_size[i]
  )
  # add OOB error to grid
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>% 
  dplyr::arrange(OOB_RMSE) %>%
  head(10)

OOB_RMSE <- vector(mode = "numeric", length = 100)
for(i in seq_along(OOB_RMSE)) {
  optimal_ranger <- ranger(
    formula         = BAM ~ ., 
    data            = file_shared, 
    num.trees       = 500,
    mtry            = 2,
    min.node.size   = 5,
    sample.fraction = .8,
    importance      = 'impurity'
  )
  OOB_RMSE[i] <- sqrt(optimal_ranger$prediction.error)
}

hist(OOB_RMSE, breaks = 20)
optimal_ranger$variable.importance %>% 
  tidy() %>%
  dplyr::arrange(desc(x)) %>%
  dplyr::top_n(5) %>%
  ggplot(aes(reorder(names, x), x)) +
  geom_col() +
  coord_flip() +
  ggtitle("Important variables")

model_ranger <- ranger(
  formula   = BAM ~ ., 
  data      = training_data, 
  num.trees = 500,
  mtry      = floor(length(features) / 3)
)

pred_ranger <- predict(model_ranger, file_shared)

file_shared$pred_ranger <- pred_ranger$predictions
ggplot(file_shared, aes(BAM, pred_ranger)) + geom_point() + geom_smooth(method = "lm")
summary(lm(BAM ~ pred_ranger, data = file_shared))
mean(abs((file_shared$BAM - file_shared$pred_ranger) / file_shared$BAM)) * 100


# For faster and parallel performance use H2O
h2o.shutdown()
h2o.no_progress()
h2o.init(max_mem_size = "8g")
y <- "BAM"
x <- setdiff(names(file_shared), y)

# turn training set into h2o object
train_h2o <- as.h2o(file_shared)

# collect the results and sort by our model performance metric of choice
hyper_grid_h2o <- list(
  ntrees      = seq(200, 500, by = 150),
  mtries      = seq(1, 4, by = 1),
  max_depth   = seq(20, 40, by = 5),
  min_rows    = seq(1, 5, by = 2),
  nbins       = seq(10, 30, by = 5),
  sample_rate = c(.55, .632, .7, .75, .8)
)

# random grid search criteria
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "mse",
  stopping_tolerance = 0.005,
  stopping_rounds = 10,
  max_runtime_secs = 30*60
)

# build grid search 
random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_grid2",
  x = x, 
  y = y, 
  training_frame = train_h2o,
  hyper_params = hyper_grid_h2o,
  search_criteria = search_criteria
)

# collect the results and sort by our model performance metric of choice
grid_perf <- h2o.getGrid(
  grid_id = "rf_grid2", 
  sort_by = "mse", 
  decreasing = FALSE
)
print(grid_perf)


# Grab the model_id for the top model, chosen by validation error
best_model_id <- grid_perf@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)

# Now let's evaluate the model performance on a test set
test_h2o <- as.h2o(testing_data)
best_model_perf <- h2o.performance(model = best_model, newdata = test_h2o)

# RMSE of best model
h2o.mse(best_model_perf) %>% sqrt()
file_h2o <- as.h2o(file_shared)

test_h2o$pred_h2o <- predict(best_model, test_h2o)
testing_data <- as.data.frame(test_h2o)
ggplot(testing_data, aes(BAM, pred_h2o)) + geom_point() + geom_smooth(method = "lm")
summary(lm(BAM ~ pred_h2o, data = testing_data))

file_h2o$pred_h2o <- predict(best_model, file_h2o)
file_shared <- as.data.frame(file_h2o)
ggplot(file_shared, aes(BAM, pred_h2o)) + geom_point() + geom_smooth(method = "lm")
summary(lm(BAM ~ pred_h2o, data = file_shared))
mean(abs((file_shared$BAM - file_shared$pred_h2o) / file_shared$BAM)) * 100

write.csv(file_shared, "RF.csv")