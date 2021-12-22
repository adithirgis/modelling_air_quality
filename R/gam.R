library(xgboost)
library(readxl)
library(tidyverse)
library(tidymodels)
library(h2o)
library(here)
library(data.table)
library(mgcv)
library(caret)

set.seed(108)

file_shared <- read_csv(here("data", "PurpleAir_BAM_Data.csv")) %>%
  select("date" = `Date and time (Before time stamp)`,
         "BAM" = `BAM-PM2.5`, "PA_CF_1" = `PA-PM2.5 (CF=1)`, "PA_CF_ATM" = `PA-PM2.5 (CF=ATM)`,
         "PA_RH" = `PA-RH`, "PA_Temp" = `PA-Temperature`, BC) %>%
  mutate(date = as.POSIXct(date, format = "%d-%m-%Y %H:%M", tz = "Asia/Kolkata"),
         hour = format(date, "%H")) %>%
  mutate_if(is.character, as.numeric) %>%
  na.omit() %>%
  select(BAM, everything(), -date, -BC, -PA_CF_1)



# file_shared <- read_csv(here("data", "PurpleAir_BAM_Data.csv")) %>%
#   select("date" = `Date and time (Before time stamp)`,
#          "BAM" = `BAM-PM2.5`, "PA_CF_1" = `PA-PM2.5 (CF=1)`, "PA_CF_ATM" = `PA-PM2.5 (CF=ATM)`,
#          "PA_RH" = `PA-RH`, "PA_Temp" = `PA-Temperature`, BC) %>%
#   mutate(date = as.POSIXct(date, format = "%d-%m-%Y %H:%M", tz = "Asia/Kolkata"),
#          hour = format(date, "%H")) %>%
#   mutate_if(is.character, as.numeric) %>%
#   na.omit() %>%
#   select(BAM, everything(), -date, -BC, -PA_CF_ATM)


# file_shared$hour <- as.factor(file_shared$hour)
# gam_model <- train(BAM ~ PA_CF_1 + PA_RH + PA_Temp, 
#            data = file_shared,
#            method = "gam",
#            trControl = trainControl(method = "cv", number = 10, 
#                                     savePredictions = TRUE)
# )
# 
# file_shared$Predicted_Corrected_1_pred <- predict(gam_model, newdata = file_shared)
# write.csv(file_shared, "gam_CF_1.csv")
# gam_cv <- as.data.frame(gam_model$pred)
# gam_cv <- gam_cv %>% 
#   subset(select == "FALSE")
# write.csv(gam_cv, "gam_CF_1_CV.csv")


gam_model <- train(BAM ~ PA_CF_ATM + PA_RH + PA_Temp, 
                   data = file_shared,
                   method = "gam",
                   trControl = trainControl(method = "cv", number = 10, 
                                            savePredictions = TRUE)
)

file_shared$Predicted_Corrected_ATM_pred <- predict(gam_model, newdata = file_shared)
write.csv(file_shared, "gam_CF_ATM.csv")
gam_cv <- as.data.frame(gam_model$pred)
gam_cv <- gam_cv %>% 
  subset(select == "FALSE")
write.csv(gam_cv, "gam_CF_ATM_CV.csv")

