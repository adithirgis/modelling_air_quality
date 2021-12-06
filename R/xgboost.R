# https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
# https://smltar.com/mlregression.html
# This is a regression problem
# Resource - https://pommevilla.rbind.io/blog/20210823_learning_shiny/
# https://docs.h2o.ai/h2o/latest-stable/h2o-r/docs/reference/h2o.xgboost.html
set.seed(007)
# For tree based 
# eta : The default value is set to 0.3. You need to specify step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features. and eta actually shrinks the feature weights to make the boosting process more conservative. The range is 0 to 1. Low eta value means model is more robust to overfitting.
# gamma : The default value is set to 0. You need to specify minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be. The range is 0 to ???. Larger the gamma more conservative the algorithm is.
# max_depth : The default value is set to 6. You need to specify the maximum depth of a tree. The range is 1 to ???.
# min_child_weight : The default value is set to 1. You need to specify the minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be. The range is 0 to ???.
# max_delta_step : The default value is set to 0. Maximum delta step we allow each tree's weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.The range is 0 to ???.
# subsample : The default value is set to 1. You need to specify the subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting. The range is 0 to 1.
# colsample_bytree : The default value is set to 1. You need to specify the subsample ratio of columns when constructing each tree. The range is 0 to 1.
# For linear
# lambda and alpha : These are regularization term on weights. Lambda default value assumed is 1 and alpha is 0.
# lambda_bias : L2 regularization term on bias and has a default value of 0.

# https://github.com/SpencerPao/Data_Science/blob/main/XGBoost/XGBoost_Regression.R


# grid_tune <- expand.grid(
#   nrounds = c(500, 1000, 1500), # number of trees
#   max_depth = c(4, 6, 8, 12, 16, 20), 
#   eta = c(0.025, 0.05, 0.1, 0.3), # Learning rate
#   gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0, 1e-8, 1e-6, 1e-4), # pruning --> Should be tuned. i.e
#   colsample_bytree = seq(0.2, 1, 0.01), # subsample ratio of columns for tree
#   min_child_weight = c(1, 2, 3), # the larger, the more conservative the model
#   subsample = seq(0.2, 1, 0.01), # used to prevent overfitting by sampling X% training
#   lambda = c(0, 0.0001, 0.001, 0.1, 1),
#   distribution = c("gaussian", "gamma"),
#   booster = c("gbtree", "gblinear", "dart")
# )


hyper_grid <- list(
  sample_rate = seq(0.2, 1, 0.01),
  reg_lambda = c(0, 0.0001, 0.001, 0.1, 1),
  reg_alpha = c(0, 0.0001, 0.001, 0.1, 1),
  col_sample_rate = seq(0.2, 1, 0.01),
  col_sample_rate_per_tree = seq(0.2, 1, 0.01),
  min_rows = 2 ^ seq(0, log2(nrow(train))-1, 1),
  min_split_improvement = c(0, 1e-8, 1e-6, 1e-4),
  ntrees = c(500, 1000, 1500), 
  max_depth = c(4, 6, 8, 12, 16, 20), 
  min_child_weight = c(1, 2, 3),
  eta = c(0.025, 0.05, 0.1, 0.3),
  gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
  distribution = c("AUTO", "gaussian", "poisson", "gamma"),
  tree_method = c("auto", "exact"),
  grow_policy = c("depthwise"),
  booster = c("gbtree", "gblinear", "dart")
)

xgb_grid <- h2o.grid(
  hyper_params = hyper_grid,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  grid_id = "xgb_grid",
  categorical_encoding = "AUTO",
  x = features,
  y = response,
  keep_cross_validation_predictions = TRUE,
  keep_cross_validation_models = TRUE,
  keep_cross_validation_fold_assignment = TRUE,
  training_frame = train,
  nfolds = 10,
  score_tree_interval = 10,
  seed = 108
)
xgb_grid

grid_perf <- h2o.getGrid(
  grid_id = "xgb_grid", 
  sort_by = "mse", 
  decreasing = FALSE
)
print(grid_perf)

best_model_id <- grid_perf@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)
best_model
h2o.varimp(best_model)
h2o.varimp_plot(best_model)
model_path <- h2o.saveModel(object = best_model, path = getwd(), force = TRUE)
print(model_path)
saved_model <- h2o.loadModel(model_path)

h2o.scoreHistory(best_model)

cv_models <- sapply(best_model@model$cross_validation_models, 
                    function(i) h2o.getModel(i$name))
cv_models


best_model_perf <- h2o.performance(model = best_model, newdata = test)
best_model_perf
h2o.mse(best_model_perf) %>% sqrt()


test$h2o_xgb <- predict(best_model, test)
test <- as.data.frame(test)
write.csv(test, "results/test_h2o_XGB.csv")

file_shared$h2o_xgb <- predict(best_model, file_shared)

model_xgb <- h2o.xgboost(x = features,
                         y = response,
                         training_frame = file_shared,
                         sample_rate = 0.81,
                         reg_lambda = 1,
                         reg_alpha = 0.1,
                         col_sample_rate = 0.9000,
                         col_sample_rate_per_tree = 0.68000,
                         min_rows = 2,
                         min_split_improvement = 0.1,
                         ntrees = 500, 
                         max_depth = 12, 
                         min_child_weight = 2,
                         eta = 0.05,
                         gamma = 0.1,
                         distribution = "gamma",
                         tree_method = "exact",
                         grow_policy = "depthwise",
                         categorical_encoding = "AUTO",
                         booster = "dart",
                         seed = 108,
                         keep_cross_validation_predictions = TRUE,
                         keep_cross_validation_models = TRUE,
                         keep_cross_validation_fold_assignment = TRUE, 
                         nfolds = 10)


model_xgb

cvpreds_id <- model_xgb@model$cross_validation_holdout_predictions_frame_id$name
file_shared$cvpreds <- h2o.getFrame(cvpreds_id)
h2o.varimp(model_xgb)
h2o.varimp_plot(model_xgb)
file_shared$h2o_xgb_m <- predict(model_xgb, file_shared)
file_shared <- as.data.frame(file_shared)
ggplot(file_shared, aes(BAM, h2o_xgb_m)) + geom_point() + geom_smooth(method = "lm")
summary(lm(BAM ~ h2o_xgb_m, data = file_shared))
mean(abs((file_shared$BAM - file_shared$h2o_xgb_m) / file_shared$BAM), na.rm = TRUE) * 100
write.csv(file_shared, "results/h2o_XGB.csv")



######################



test$h2o_xgb <- predict(best_model, test)
test <- as.data.frame(test)
write.csv(test, "results/test_h2o_XGB_CF_1.csv")

file_shared$h2o_xgb <- predict(best_model, file_shared)

model_xgb <- h2o.xgboost(x = features,
                         y = response,
                         training_frame = file_shared,
                         sample_rate = 0.81,
                         reg_lambda = 1,
                         reg_alpha = 0.1,
                         col_sample_rate = 0.9000,
                         col_sample_rate_per_tree = 0.68000,
                         min_rows = 2,
                         min_split_improvement = 0.1,
                         ntrees = 500, 
                         max_depth = 12, 
                         min_child_weight = 2,
                         eta = 0.05,
                         gamma = 0.1,
                         distribution = "gamma",
                         tree_method = "exact",
                         grow_policy = "depthwise",
                         categorical_encoding = "AUTO",
                         booster = "dart",
                         seed = 108,
                         keep_cross_validation_predictions = TRUE,
                         keep_cross_validation_models = TRUE,
                         keep_cross_validation_fold_assignment = TRUE, 
                         nfolds = 10)


model_xgb

cvpreds_id <- model_xgb@model$cross_validation_holdout_predictions_frame_id$name
file_shared$cvpreds <- h2o.getFrame(cvpreds_id)
h2o.varimp(model_xgb)
h2o.varimp_plot(model_xgb)
file_shared$h2o_xgb_m <- predict(model_xgb, file_shared)
file_shared <- as.data.frame(file_shared)
ggplot(file_shared, aes(BAM, h2o_xgb_m)) + geom_point() + geom_smooth(method = "lm")
summary(lm(BAM ~ h2o_xgb_m, data = file_shared))
mean(abs((file_shared$BAM - file_shared$h2o_xgb_m) / file_shared$BAM), na.rm = TRUE) * 100
write.csv(file_shared, "results/h2o_XGB_CF_1.csv")




















grid_tune <- expand.grid(
  nrounds = c(500, 1000), # number of trees
  max_depth = c(4, 6, 8, 12, 16), 
  eta = c(0.025, 0.1, 0.3), # Learning rate
  gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0, 1e-4), # pruning --> Should be tuned. i.e
  colsample_bytree = seq(0.2, 1, 0.2), # subsample ratio of columns for tree
  min_child_weight = c(1, 2, 3), # the larger, the more conservative the model
  subsample = seq(0.2, 1, 0.2), # used to prevent overfitting by sampling X% training
  lambda = c(0, 0.001, 0.1, 1)
)
xgb_train_rmse <- NULL
xgb_test_rmse <- NULL

start_t <- Sys.time()
for (j in 1:nrow(grid_tune)) {
  set.seed(108)
  m_xgb_untuned <- xgb.cv(
    data = data.matrix(tidy_train[, 2:5]),
    label = data.matrix(tidy_train[, 1]),
    nrounds = grid_tune$nrounds[j],
    objective = "reg:squarederror",
    nfold = 10,
    colsample_bytree = grid_tune$colsample_bytree[j],
    min_child_weight = grid_tune$min_child_weight[j],
    lambda = grid_tune$lambda[j],
    subsample = grid_tune$subsample[j],
    gamma = grid_tune$gamma[j],
    max_depth = grid_tune$max_depth[j],
    eta = grid_tune$eta[j]
  )
  
  xgb_train_rmse[j] <- m_xgb_untuned$evaluation_log$train_rmse_mean[m_xgb_untuned$best_iteration]
  xgb_test_rmse[j] <- m_xgb_untuned$evaluation_log$test_rmse_mean[m_xgb_untuned$best_iteration]
  cat(j, "\n")
}
end_t <- Sys.time()

#ideal hyperparamters
ideal_para <- grid_tune[which.min(xgb_test_rmse), ]

xgb_model <-
  xgboost(
    data = data.matrix(tidy_train[, 2:5]),
    label = data.matrix(tidy_train[, 1]),
    nrounds = ideal_para$nrounds,
    objective = "reg:squarederror",
    max_depth = ideal_para$max_depth,
    eta = ideal_para$eta,
    colsample_bytree = ideal_para$colsample_bytree,
    min_child_weight = ideal_para$min_child_weight,
    lambda = ideal_para$lambda,
    subsample = ideal_para$subsample,
    gamma = ideal_para$gamma
  )


tidy_test$xgb <- predict(xgb_model, tidy_test[, 2:5])
tidy_test <- as.data.frame(tidy_test)

file_shared$xgb <- predict(xgb_model, file_shared[, 2:5])
file_shared <- as.data.frame(file_shared)

ggplot(file_shared, aes(BAM, xgb)) + geom_point() + geom_smooth(method = "lm")
summary(lm(BAM ~ xgb, data = file_shared))
mean(abs((file_shared$BAM - file_shared$xgb) / file_shared$BAM)) * 100
write.csv(file_shared, "xgboost.csv")