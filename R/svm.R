# https://stackoverflow.com/questions/60134569/r-apply-svm-function-for-group-by-in-data-frame
# https://www.kdnuggets.com/2017/03/building-regression-models-support-vector-regression.html
# https://d2l.ai/chapter_preface/index.html

# function of svm
# svm_ftn <- function(BAM, PA_CF_ATM, PA_RH, PA_Temp, hour, file_shared) {
#   svm_model <- svm(BAM ~ PA_CF_ATM + PA_RH + PA_Temp + hour, file_shared)
#   file_shared$predicted_PA <- predict(svm_model, file_shared)
# }

# ? cost parameter to avoid overfit and error 
# SVR technique relies on kernel functions to construct the model. 
# The commonly used kernel functions are: a) Linear, b) Polynomial, c) Sigmoid and d) Radial Basis. 
# While implementing SVR technique, the user needs to select the appropriate kernel function.  
# The selection of kernel function is a tricky and requires optimization techniques for the best selection.
# svm function in R considers maximum allowed error (??i) to be 0.1.
# In order to avoid over-fitting, the svm SVR function allows us to penalize the regression through cost function. 
# The SVR technique is flexible in terms of maximum allowed error and penalty cost.
# This process of searching for the best model is called tuning of SVR model.
# The tune function evaluates performance of 1100 models (100*11) i.e. for every 
# combination of maximum allowable error (0 , 0.1 , . . . . . 1) and cost parameter (1 , 2 , 3 , 4 , 5 , . . . . . 100).

# Tuning SVR model by varying values of maximum allowable error and cost parameter

# Takes a lot of time, any chance to reduce the time taken
# Tune the SVM model

all_cores <- parallelly::availableCores()
future::plan("multisession", workers = all_cores) # on Windows
tune_model_svm <- tune.svm(BAM ~ ., data = tidy_train, cost = 1:100, elsilon = seq(0, 1, 0.1))

# Print optimum value of parameters
print(tune_model_svm)

# Plot the perfrormance of SVM Regression model
plot(tune_model_svm)

# The best model is the one with lowest MSE. 
# The darker the region the lower the MSE, which means better the model.

# Find out the best model
best_svm_model <- tune_model_svm$best.model

# Predict Y using best model
file_shared$pred_svm_new <- predict(best_svm_model, file_shared)

# Calculate RMSE of the best model 
ggplot(file_shared, aes(BAM, pred_svm_new)) + geom_point() + geom_smooth(method = "lm")
summary(lm(BAM ~ pred_svm_new, data = file_shared))
mean(abs((file_shared$BAM - file_shared$pred_svm_new) / file_shared$BAM)) * 100


# Calculate parameters of the Best SVR model
# Find value of W
W <- t(best_svm_model$coefs) %*% best_svm_model$SV
# Find value of b
b <- best_svm_model$rho

write.csv(file_shared, "svm.csv")