library(LiblineaR)
library(readxl)
library(xgboost)
library(doParallel)
library(tidyverse)
library(e1071)
library(caret)
library(keras)
library(here)
library(reticulate)
library(tidymodels)
library(tfdatasets)
library(future)
library(ranger)
library(parallelly)
library(mgcv)
library(neuralnet)
library(reticulate)


set.seed(108)
# use_python("/usr/local/bin/python")
# use_virtualenv("myenv")
# os <- import("os")
# os$listdir(".")

file_shared <- read_csv(here("data", "PurpleAir_BAM_Data.csv")) %>%
  select("date" = `Date and time (Before time stamp)`, 
         "BAM" = `BAM-PM2.5`, "PA_CF_1" = `PA-PM2.5 (CF=1)`, "PA_CF_ATM" = `PA-PM2.5 (CF=ATM)`, 
         "PA_RH" = `PA-RH`, "PA_Temp" = `PA-Temperature`, BC) %>%
  mutate(date = as.POSIXct(date, format = "%d-%m-%Y %H:%M", tz = "Asia/Kolkata"),
         hour = format(date, "%H")) %>% 
  mutate_if(is.character, as.numeric) %>% 
  na.omit() %>%
  select(BAM, everything(), -date, -BC, -PA_CF_1)

file_shared$hour <- as.factor(file_shared$hour)

# Let's start by splitting the data into a testing and training set:
tidy_split <- initial_split(file_shared)
tidy_train <- training(tidy_split)
tidy_test <- testing(tidy_split)
tidy_kfolds <- vfold_cv(tidy_train)