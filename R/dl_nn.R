hyper_grid <- list(
  activation = c("Tanh", "TanhWithDropout"),
  hidden = list(c(5, 5, 5, 5, 5), c(30, 30, 30, 30), c(50, 50, 50, 50), c(100, 100, 100, 100)),
  epochs = c(50, 100, 200, 300, 400, 500),
  l1 = c(0, 0.00001, 0.0001, 1e-5),
  l2 = c(0, 0.00001, 0.0001),
  rate = c(0, 0.005, 0.001),
  rate_annealing = c(1e-7, 1e-6),
  rho = c(0.9, 0.95, 0.99, 0.999),
  epsilon = c(1e-10, 1e-8, 1e-6, 1e-4),
  momentum_start = c(0, 0.5),
  momentum_stable = c(0.99, 0.5, 0),
  distribution = c("AUTO"),
  input_dropout_ratio = c(0, 0.1, 0.2),
  max_w2 = c(1, 10, 100)
)

dl_grid <- h2o.grid(algorithm = "deeplearning", 
                    x = features,
                    y = response,
                    grid_id = "dl_grid",
                    training_frame = train,
                    keep_cross_validation_predictions = TRUE,
                    keep_cross_validation_models = TRUE,
                    keep_cross_validation_fold_assignment = TRUE, 
                    nfolds = 10,                           
                    hyper_params = hyper_grid,
                    search_criteria = search_criteria,
                    seed = 108
)
dl_grid

grid_perf <- h2o.getGrid(
  grid_id = "dl_grid", 
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
plot(best_model, 
     timestep = "epochs", 
     metric = "rmse")

cv_models <- sapply(best_model@model$cross_validation_models, 
                    function(i) h2o.getModel(i$name))
cv_models

plot(cv_models[[1]], 
     timestep = "epochs", 
     metric = "rmse")

best_model_perf <- h2o.performance(model = best_model, newdata = test)
best_model_perf
h2o.mse(best_model_perf) %>% sqrt()







test$h2o_DL <- predict(best_model, test)
test <- as.data.frame(test)
write.csv(test, "results/test_h2o_DL.csv")

file_shared$h2o_dl <- predict(best_model, file_shared)


model_dl <- h2o.deeplearning(x = features,
                             y = response,
                             training_frame = file_shared,
                             distribution = "gaussian",
                             hidden =  c(50, 50, 50, 50),
                             rate = 0.000,
                             epochs = 19,
                             l1 = 0.000, 
                             l2 = 0.0001,
                             rho = 0.95,
                             categorical_encoding = "AUTO",
                             epsilon = 1e-7,
                             momentum_start = 0,
                             input_dropout_ratio = 0.1,
                             max_w2 = 100,
                             reproducible = TRUE,
                             activation = "Rectifier",
                             seed = 108,
                             keep_cross_validation_predictions = TRUE,
                             keep_cross_validation_models = TRUE,
                             keep_cross_validation_fold_assignment = TRUE, 
                             nfolds = 10)

model_dl

cvpreds_id <- model_dl@model$cross_validation_holdout_predictions_frame_id$name
file_shared$cvpreds <- h2o.getFrame(cvpreds_id)

h2o.varimp(model_dl)
h2o.varimp_plot(model_dl)

file_shared$h2o_dl_m <- predict(model_dl, file_shared)
file_shared <- as.data.frame(file_shared)
ggplot(file_shared, aes(BAM, h2o_dl_m)) + geom_point() + geom_smooth(method = "lm")
summary(lm(BAM ~ h2o_dl_m, data = file_shared))
mean(abs((file_shared$BAM - file_shared$h2o_dl_m) / file_shared$BAM), na.rm = TRUE) * 100
write.csv(file_shared, "results/h2o_DL.csv")





#########

test$h2o_DL <- predict(best_model, test)
test <- as.data.frame(test)
write.csv(test, "results/test_h2o_DL_CF_1.csv")

file_shared$h2o_dl <- predict(best_model, file_shared)


model_dl <- h2o.deeplearning(x = features,
                             y = response,
                             training_frame = file_shared,
                             distribution = "gaussian",
                             hidden =  c(30, 30, 30, 30),
                             rate = 0.000,
                             epochs = 207,
                             l1 = 0.00001, 
                             l2 = 0.00001,
                             rho = 0.90,
                             categorical_encoding = "AUTO",
                             epsilon = 1e-7,
                             momentum_start = 0,
                             input_dropout_ratio = 0.2,
                             max_w2 = 10,
                             reproducible = TRUE,
                             activation = "Tanh",
                             seed = 108,
                             keep_cross_validation_predictions = TRUE,
                             keep_cross_validation_models = TRUE,
                             keep_cross_validation_fold_assignment = TRUE, 
                             nfolds = 10)

model_dl

cvpreds_id <- model_dl@model$cross_validation_holdout_predictions_frame_id$name
file_shared$cvpreds <- h2o.getFrame(cvpreds_id)

h2o.varimp(model_dl)
h2o.varimp_plot(model_dl)

file_shared$h2o_dl_m <- predict(model_dl, file_shared)
file_shared <- as.data.frame(file_shared)
ggplot(file_shared, aes(BAM, h2o_dl_m)) + geom_point() + geom_smooth(method = "lm")
summary(lm(BAM ~ h2o_dl_m, data = file_shared))
mean(abs((file_shared$BAM - file_shared$h2o_dl_m) / file_shared$BAM), na.rm = TRUE) * 100
write.csv(file_shared, "results/h2o_DL_CF_1.csv")


