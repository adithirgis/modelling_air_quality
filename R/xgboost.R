# https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
# https://smltar.com/mlregression.html
# This is a regression problem
# Resource - https://pommevilla.rbind.io/blog/20210823_learning_shiny/
# https://docs.h2o.ai/h2o/latest-stable/h2o-r/docs/reference/h2o.xgboost.html
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

set.seed(007)
# xgboost
hyper_grid <- list(
  sample_rate = seq(0.2, 1, 0.01),
  reg_lambda = c(0, 0.0001, 0.001, 0.1, 1),
  reg_alpha = c(0, 0.0001, 0.001, 0.1, 1),
  col_sample_rate = seq(0.2, 1, 0.01),
  col_sample_rate_per_tree = seq(0.2, 1, 0.01),
  min_rows = 2 ^ seq(0, log2(nrow(train))-1, 1),
  min_split_improvement = c(0, 1e-8, 1e-6, 1e-4),
  ntrees = c(500, 1000, 1500), 
  max_depth = c(4, 6, 8, 12, 16, 20), min_child_weight = c(1, 2, 3),
  eta = c(0.025, 0.05, 0.1, 0.3),
  gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
  distribution = c("AUTO", "gaussian", "poisson", "gamma"),
  tree_method = c("auto", "exact", "approx"),
  grow_policy = c("depthwise"),
  booster = c("gbtree", "gblinear", "dart")
)


grid <- h2o.grid(
  hyper_params = hyper_grid,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  grid_id = "xgb_grid",
  x = features,
  y = response,
  training_frame = train,
  nfolds = 10,
  score_tree_interval = 10,
  seed = 108
)

grid_perf <- h2o.getGrid(
  grid_id = "xgb_grid", 
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

test$h2o_DRF <- predict(best_model, test)
test <- as.data.frame(test)


file_shared$h2o_XGB <- predict(best_model, file_shared)
file_shared <- as.data.frame(file_shared)


ggplot(file_shared, aes(PM2.5, h2o_XGB)) + geom_point() + geom_smooth(method = "lm")
summary(lm(PM2.5 ~ h2o_XGB, data = file_shared))
mean(abs((file_shared$PM2.5 - file_shared$h2o_XGB) / file_shared$PM2.5), na.rm = TRUE) * 100
write.csv(file_shared, "h2o_XGB.csv")

