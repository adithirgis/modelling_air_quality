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
  ntrees      = seq(100, 500, by = 100),
  mtries      = seq(2, 10, by = 2),
  max_depth   = seq(20, 40, by = 5),
  min_rows    = seq(1, 5, by = 2),
  nbins       = seq(10, 40, by = 5),
  sample_rate = c(.55, .632, 0.7, .75, .8)
)

# build grid search 

start <- Sys.time()
random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_grid",
  x = features, 
  y = response, 
  training_frame = train,
  nfolds = 10,  
  hyper_params = hyper_grid,
  search_criteria = search_criteria
)
end <- Sys.time() 
beepr::beep()

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

# Save model
model_path <- h2o.saveModel(object = best_model, path = getwd(), force = TRUE)
print(model_path)
saved_model <- h2o.loadModel(model_path)

h2o.scoreHistory(best_model)

# Get the CV models from the `best_model` object
cv_models <- sapply(best_model@model$cross_validation_models, 
                    function(i) h2o.getModel(i$name))

# model_path <- h2o.saveModel(object = cv_models, path = getwd(), force = TRUE)
# print(model_path)

# Plot the scoring history over time
plot(cv_models[[1]], 
     timestep = "epochs", 
     metric = "rmse")

# Now let's evaluate the model performance on a test set
best_model_perf <- h2o.performance(model = best_model, newdata = test)

# RMSE of best model
h2o.mse(best_model_perf) %>% sqrt()

test$h2o_DRF <- predict(best_model, test)
test <- as.data.frame(test)

file_shared$h2o_DRF <- predict(best_model, file_shared)
file_shared <- as.data.frame(file_shared)
ggplot(file_shared, aes(PM2.5, h2o_DRF)) + geom_point() + geom_smooth(method = "lm")
summary(lm(PM2.5 ~ h2o_DRF, data = file_shared))
mean(abs((file_shared$PM2.5 - file_shared$h2o_DRF) / file_shared$PM2.5), na.rm = TRUE) * 100

write.csv(file_shared, "h2o_DRF.csv")

