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

