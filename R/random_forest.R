# https://uc-r.github.io/random_forests

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

# `ntree`: number of trees. We want enough trees to stabilize the error but using too many trees is unnecessarily inefficient, especially when using large data sets.
# `mtry`: the number of variables to randomly sample as candidates at each split. When mtry = p the model equates to bagging. When mtry = 1 the split variable is completely random, so all variables get a chance but can lead to overly biased results. A common suggestion is to start with 5 values evenly spaced across the range from 2 to p.
# `sampsize`: the number of samples to train on. The default value is 63.25% of the training set since this is the expected value of unique observations in the bootstrap sample. Lower sample sizes can reduce the training time but may introduce more bias than necessary. Increasing the sample size can increase performance but at the risk of overfitting because it introduces more variance. Typically, when tuning this parameter we stay near the 60-80% range.
# `nodesize`: minimum number of samples within the terminal nodes. Controls the complexity of the trees. Smaller node size allows for deeper, more complex trees and smaller node results in shallower trees. This is another bias-variance tradeoff where deeper trees introduce more variance (risk of overfitting) and shallower trees introduce more bias (risk of not fully capturing unique patters and relatonships in the data).
# `maxnodes`: maximum number of terminal nodes. Another way to control the complexity of the trees. More nodes equates to deeper, more complex trees and less nodes result in shallower trees.



# collect the results and sort by our model performance metric of choice
hyper_grid <- list(
  ntrees      = seq(200, 400, by = 200),
  mtries      = seq(2, 6, by = 4),
  max_depth   = seq(10, 40, by = 20),
  min_rows    = seq(5, 10, by = 5),
  nbins       = seq(10, 40, by = 20),
  sample_rate = c(.55, .632, .75, .8)
)

# Find Parameters: Use Hyper Parameter Tuning on a "Training Dataset" that sections your training data into 5-Folds. The output at Stage 1 is the parameter set.
# build grid search 
random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_grid",
  x = features, 
  y = response, 
  training_frame = train,
  keep_cross_validation_predictions = TRUE,
  keep_cross_validation_models = TRUE,
  keep_cross_validation_fold_assignment = TRUE, 
  nfolds = 10,  
  categorical_encoding = "AUTO",
  hyper_params = hyper_grid,
  search_criteria = search_criteria
)

random_grid

# collect the results and sort by our model performance metric of choice
grid_perf <- h2o.getGrid(
  grid_id = "rf_grid", 
  sort_by = "mse", 
  decreasing = FALSE
)
print(grid_perf)


# Grab the model_id for the top model, chosen by validation error
best_model_id <- grid_perf@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)
best_model

h2o.varimp(best_model)
h2o.varimp_plot(best_model)


# Save model
model_path <- h2o.saveModel(object = best_model, path = getwd(), force = TRUE)
print(model_path)
saved_model <- h2o.loadModel(model_path)

h2o.scoreHistory(best_model)

# Get the CV models from the `best_model` object
cv_models <- sapply(best_model@model$cross_validation_models, 
                    function(i) h2o.getModel(i$name))
cv_models

# model_path <- h2o.saveModel(object = cv_models, path = getwd(), force = TRUE)
# print(model_path)


# Compare and Select Best Model: Evaluate the performance on a hidden "Test Dataset". The ouput at Stage 2 is that we determine best model.
# Now let's evaluate the model performance on a test set
best_model_perf <- h2o.performance(model = best_model, newdata = test)
best_model_perf
# RMSE of best model
h2o.mse(best_model_perf) %>% sqrt()


test$h2o_rf <- predict(best_model, test)
test <- as.data.frame(test)
write.csv(test, "results/test_h2o_RF.csv")

file_shared$h2o_rf <- predict(best_model, file_shared)

# Train Final Model: Once we have selected the best model, we train on the full dataset. This model goes into production.
model_drf <- h2o.randomForest(x = features, 
                              y = response, 
                              training_frame = file_shared,
                              ntrees = 200,
                              sample_rate = 0.8,
                              max_depth = 10,
                              min_rows = 5,
                              nbins = 10,
                              keep_cross_validation_predictions = TRUE,
                              keep_cross_validation_models = TRUE,
                              keep_cross_validation_fold_assignment = TRUE, 
                              nfolds = 10)

model_drf


cvpreds_id <- model_drf@model$cross_validation_holdout_predictions_frame_id$name
file_shared$cvpreds <- h2o.getFrame(cvpreds_id)

h2o.varimp(model_drf)
h2o.varimp_plot(model_drf)
file_shared$h2o_rf_m <- predict(model_drf, file_shared)
file_shared <- as.data.frame(file_shared)
ggplot(file_shared, aes(BAM, h2o_rf_m)) + geom_point() + geom_smooth(method = "lm")
summary(lm(BAM ~ h2o_rf_m, data = file_shared))
mean(abs((file_shared$BAM - file_shared$h2o_rf_m) / file_shared$BAM), na.rm = TRUE) * 100

write.csv(file_shared, "results/h2o_RF.csv")



###################

test$h2o_rf <- predict(best_model, test)
test <- as.data.frame(test)
write.csv(test, "results/test_h2o_RF_CF_1.csv")

file_shared$h2o_rf <- predict(best_model, file_shared)

# Train Final Model: Once we have selected the best model, we train on the full dataset. This model goes into production.
model_drf <- h2o.randomForest(x = features, 
                              y = response, 
                              training_frame = file_shared,
                              ntrees = 200,
                              sample_rate = 0.8,
                              max_depth = 10,
                              min_rows = 5,
                              nbins = 10,
                              keep_cross_validation_predictions = TRUE,
                              keep_cross_validation_models = TRUE,
                              keep_cross_validation_fold_assignment = TRUE, 
                              nfolds = 10)

model_drf


cvpreds_id <- model_drf@model$cross_validation_holdout_predictions_frame_id$name
file_shared$cvpreds <- h2o.getFrame(cvpreds_id)

h2o.varimp(model_drf)
h2o.varimp_plot(model_drf)
file_shared$h2o_rf_m <- predict(model_drf, file_shared)
file_shared <- as.data.frame(file_shared)
ggplot(file_shared, aes(BAM, h2o_rf_m)) + geom_point() + geom_smooth(method = "lm")
summary(lm(BAM ~ h2o_rf_m, data = file_shared))
mean(abs((file_shared$BAM - file_shared$h2o_rf_m) / file_shared$BAM), na.rm = TRUE) * 100

write.csv(file_shared, "results/h2o_RF_CF_1.csv")











hyper_grid <- expand.grid(
  mtry       = seq(1, 4, by = 1),
  node_size  = seq(2, 6, by = 2),
  sampe_size = c(.55, .632, .70, .80),
  num.trees = seq(100, 500, by = 100),
  OOB_RMSE   = 0
)
nrow(hyper_grid)


for(i in 1:nrow(hyper_grid)) {
  model <- ranger(
    formula         = BAM ~ ., 
    data            = tidy_train, 
    num.trees       = hyper_grid$num.trees[i],
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sampe_size[i],
    seed            = 007
  )
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>% 
  dplyr::arrange(OOB_RMSE) %>%
  head(10)

# Check the best model and add it here 
OOB_RMSE <- vector(mode = "numeric", length = 100)

for(i in seq_along(OOB_RMSE)) {
  
  optimal_ranger <- ranger(
    formula         = BAM ~ ., 
    data            = tidy_train, 
    num.trees       = 100,
    mtry            = 4,
    min.node.size   = 2,
    sample.fraction = 0.7,
    importance      = 'impurity'
  )
  
  OOB_RMSE[i] <- sqrt(optimal_ranger$prediction.error)
}

hist(OOB_RMSE, breaks = 20)

# Check the best model and add it here 
best_model <- ranger(
  formula         = BAM ~ ., 
  data            = tidy_train, 
  num.trees       = 100,
  mtry            = 4,
  min.node.size   = 2,
  sample.fraction = 0.7,
  seed            = 7
)

pred <- predict(best_model, tidy_test)
tidy_test$RF <- pred$predictions

pred <- predict(best_model, file_shared)
file_shared$RF <- pred$predictions

ggplot(file_shared, aes(BAM, RF)) + geom_point() + geom_smooth(method = "lm")
summary(lm(BAM ~ RF, data = file_shared))
mean(abs((file_shared$BAM - file_shared$RF) / file_shared$BAM), na.rm = TRUE) * 100

write.csv(file_shared, "RF.csv")





