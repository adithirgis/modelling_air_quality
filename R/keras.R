# https://www.datacamp.com/community/tutorials/keras-r-deep-learning
# https://rpubs.com/juanhklopper/dnn_for_regression - Should I use this?

# install.packages("keras")
# keras::install_keras()

use_python("/usr/local/bin/python")
use_virtualenv("myenv")
os <- import("os")
os$listdir(".")

# https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_regression/
spec <- feature_spec(tidy_train, BAM ~ . ) %>% 
  step_numeric_column(all_numeric(), normalizer_fn = scaler_standard()) %>% 
  fit()

spec

layer <- layer_dense_features(
  feature_columns = dense_features(spec), 
  dtype = tf$float32
)
layer(tidy_train)


input <- layer_input_from_dataset(tidy_train %>% select(-BAM))

output <- input %>% 
  layer_dense_features(dense_features(spec)) %>% 
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1) 

model <- keras_model(input, output)

summary(model)

model %>% 
  compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mean_absolute_error")
  )


build_model <- function() {
  input <- layer_input_from_dataset(tidy_train %>% select(-BAM))
  
  output <- input %>% 
    layer_dense_features(dense_features(spec)) %>% 
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1) 
  
  model <- keras_model(input, output)
  
  model %>% 
    compile(
      loss = "mse",
      optimizer = optimizer_rmsprop(),
      metrics = list("mean_absolute_error")
    )
  
  model
}


print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

model <- build_model()
# This takes time ~ 3 mins
history <- model %>% fit(
  x = tidy_train %>% select(-BAM),
  y = tidy_train$BAM,
  epochs = 500,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(print_dot_callback)
)


plot(history)

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

model <- build_model()

history <- model %>% fit(
  x = tidy_train %>% select(-BAM),
  y = tidy_train$BAM,
  epochs = 500,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(early_stop)
)

plot(history)

c(loss, mae) %<-% (model %>% evaluate(tidy_test %>% select(-BAM), tidy_test$BAM, verbose = 0))

paste0("Mean absolute error on test set: ", sprintf("%.2f", mae))

file_shared$pred_keras <- model %>% predict(tidy_test %>% select(-BAM))
ggplot(file_shared, aes(BAM, pred_keras)) + geom_point() + geom_smooth(method = "lm")
summary(lm(BAM ~ pred_keras, data = file_shared))
mean(abs((file_shared$BAM - file_shared$pred_keras) / file_shared$BAM)) * 100

write.csv(file_shared, "keras.csv")

# save_model_weights_hdf5("my_model_weights.h5")
# model %>% load_model_weights_hdf5("my_model_weights.h5")

