
gam_model <- gam(BAM ~ s(PA_CF_1) + s(PA_RH) + s(PA_Temp) + s(hour), data = file_shared)

file_shared$Predicted_Corrected_PM25_PA_both <- predict(gam_model, newdata = file_shared)

write.csv(file_shared, "gam.csv")

