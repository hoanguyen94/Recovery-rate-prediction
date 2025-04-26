install.packages("pacman")

require("pacman")

# Use pacman to load add-on packages as desired
pacman::p_load(pacman, rio, tidyverse, caret, Cubist, Metrics, ggplot2, openxlsx) 

# CSV
current_path = getwd()
train_features_file = paste(current_path, "/data/train_features2.xlsx", sep = "")
train_labels_file = paste(current_path, "/data/train_labels2.xlsx", sep = "")
test_features_file = paste(current_path, "/data/test_features2.xlsx", sep = "")
test_labels_file = paste(current_path, "/data/test_labels2.xlsx", sep = "")

training_prediction_file = paste(current_path, "/output/train_predictions.xlsx", sep = "")
test_prediction_file = paste(current_path, "/output/test_predictions.xlsx", sep = "")
sheet_name = "cubist"

set.seed(42)

train_features = import(train_features_file)
train_labels = import(train_labels_file)
test_features = import(test_features_file)
test_labels = import(test_labels_file)

names(test_features) <- gsub(" ", "_", names(test_features))
names(train_features) <- gsub(" ", "_", names(train_features))

train_labels = train_labels$rr1_30
test_labels = test_labels$rr1_30
head(train_features)

dim(train_features)

summary(train_labels)
summary(test_labels)

train_features = train_features %>%
  mutate(across(where(is.logical), ~as.integer(.)))

test_features = test_features %>%
  mutate(across(where(is.logical), ~as.integer(.)))

# split dataset
set.seed(42)
# test_size = 0.25

# labels = df$rr1_30
# features = subset(df, select = -c(rr1_7, rr2_7, rr1_30, rr2_30, ID_BB_COMPANY, bond_isin))

# test_index = createDataPartition(labels, p = 0.25, list = FALSE)
# test_labels = labels[test_index]
# training_labels = labels[-test_index]

# test_features = features[test_index, ]
# training_features = features[-test_index, ]

NROW(test_labels)
NROW(train_labels)
dim(test_features)
dim(train_features)

###### MODEL TUNING
grid = expand.grid(committees = c(1, 10, 50, 100), neighbors = c(0, 1, 5, 9))

caret_grid = train(
  x = train_features, 
  y = train_labels,
  method = "cubist",
  tuneGrid = grid,
  trControl = trainControl(method = "cv")
)
summary(caret_grid)

caret_grid

# calculate training metrics
train_prediction = predict(caret_grid, train_features)
training_mae = mae(train_labels, train_prediction)
training_rmse = rmse(train_labels, train_prediction)
training_mape = mape(train_labels, train_prediction)
training_r_squared = cor(train_labels, train_prediction)^2


# calculate test metrics
test_prediction = predict(caret_grid, test_features)
test_mae = mae(test_labels, test_prediction)
test_rmse = rmse(test_labels, test_prediction)
test_mape = mape(test_labels, test_prediction)
test_r_squared = cor(test_labels, test_prediction)^2


### MODEL RUNNING

model_rules = cubist(x = train_features, y = train_labels, committees = 100)
summary(model_rules)

# calculate training metrics
train_prediction = predict(model_rules, train_features, neighbors = 9)
training_mae = mae(train_labels, train_prediction)
training_rmse = rmse(train_labels, train_prediction)
training_mape = mape(train_labels, train_prediction)
training_r_squared = cor(train_labels, train_prediction)^2


# calculate test metrics
test_prediction = predict(model_rules, test_features, neighbors = 9)
test_mae = mae(test_labels, test_prediction)
test_rmse = rmse(test_labels, test_prediction)
test_mape = mape(test_labels, test_prediction)
test_r_squared = cor(test_labels, test_prediction)^2

training_df = data.frame(value = train_prediction)
test_df = data.frame(value = test_prediction)

###### FEATURE IMPORTANCES
importances = caret::varImp(model_rules)

# Add feature names as a new column
importances$Feature = rownames(importances)

# Sort by importance in descending order and select top 50
top_features = importances[order(-importances$Overall), ][1:50, ]

# Plot using ggplot2
plot <- ggplot(top_features, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_col(fill = "steelblue", width = 0.5) +
  coord_flip() +  # Flip the axes to make the plot horizontal
  labs(x = "Feature", y = "Importance", title = "Top 50 Feature Importances") +
  theme_minimal()  # Clean theme

# Save the plot
ggsave(
    filename = "feature_importance.png",
    plot = plot,
    path = paste(current_path, "/output/cubist", sep = ""),
    width = 10,
    height = 8
)



##### TESTING WITH FEATURE SELECTION

train_features_small = train_features %>% select(top_features$Feature)
test_features_small = test_features %>% select(top_features$Feature)


model_rules_small = cubist(x = train_features_small, y = train_labels, committees = 100)
summary(model_rules_small)

# calculate training metrics
train_prediction = predict(model_rules_small, train_features_small, neighbors = 9)
training_mae = mae(train_labels, train_prediction)
training_rmse = rmse(train_labels, train_prediction)
training_mape = mape(train_labels, train_prediction)
training_r_squared = cor(train_labels, train_prediction)^2


# calculate test metrics
test_prediction = predict(model_rules_small, test_features_small, neighbors = 9)
test_mae = mae(test_labels, test_prediction)
test_rmse = rmse(test_labels, test_prediction)
test_mape = mape(test_labels, test_prediction)
test_r_squared = cor(test_labels, test_prediction)^2

# save predictions
train_df = data.frame(predictions = train_prediction)
train_wb <- loadWorkbook(training_prediction_file)
addWorksheet(train_wb, sheetName = sheet_name)
writeData(train_wb, sheet_name, train_df)
saveWorkbook(train_wb, training_prediction_file, overwrite = TRUE)

test_df = data.frame(predictions = test_prediction)
test_wb <- loadWorkbook(test_prediction_file)
addWorksheet(test_wb, sheetName = sheet_name)
writeData(test_wb, sheet_name, test_df)
saveWorkbook(test_wb, test_prediction_file, overwrite = TRUE)

###### FEATURE IMPORTANCES
importances_small = caret::varImp(model_rules_small)

# Add feature names as a new column
importances_small$Feature = rownames(importances_small)

# Sort by importance in descending order and select top 50
top_features = importances_small[order(-importances_small$Overall), ]

# Plot using ggplot2
ggplot(top_features, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_col(fill = "steelblue", width = 0.5) +
  coord_flip() +  # Flip the axes to make the plot horizontal
  labs(x = "Feature", y = "Importance", title = "Top 50 Feature Importances") +
  theme_minimal()  # Clean theme

# model tuning
grid = expand.grid(committees = c(1, 10, 50, 100), neighbors = c(0, 1, 5, 9))

caret_grid = train(
  x = train_features_small, 
  y = train_labels,
  method = "cubist",
  tuneGrid = grid,
  trControl = trainControl(method = "cv")
)
summary(caret_grid)

caret_grid

# CV
features = rbind(train_features, test_features)
labels = c(train_labels, test_labels)

# Define the train control with 5-fold cross-validation
train_control <- trainControl(method = "cv", 
                              number = 5, 
                              verboseIter = TRUE, 
                              returnResamp = "all")

# set specific values from grid search
grid <- expand.grid(committees = 100, neighbors = 9)

caret_grid = train(
  x = features, 
  y = labels,
  method = "cubist",
  tuneGrid = grid,
  trControl = train_control
)
summary(caret_grid)

caret_grid

### SHAPLEY VALUES
# Install and load required packages
if (!require(fastshap)) install.packages("fastshap")
library(fastshap)
library(ggplot2)

# Create a prediction wrapper function for the Cubist model
cubist_pred <- function(model, newdata) {
    as.numeric(predict(model, newdata, neighbors = 9))  # using neighbors=9 as in your original code
}

# Calculate SHAP values for the model
# Using your best model (model_rules) and training data
set.seed(42)  # for reproducibility

shap_values <- fastshap::explain(
    model_rules,  # your Cubist model
    X = as.data.frame(train_features),  # feature data
    pred_wrapper = cubist_pred,
    nsim = 100,  # number of Monte Carlo simulations
)

# Create summary plot function
plot_shap_summary <- function(shap_values, feature_data) {
    # Convert SHAP values to long format
    shap_long <- as.data.frame(shap_values) %>%
        gather(key = "feature", value = "shap_value") %>%
        mutate(
            abs_shap = abs(shap_value),
            feature = factor(feature)
        )
    
    # Calculate mean absolute SHAP value for each feature
    feature_importance <- shap_long %>%
        group_by(feature) %>%
        summarise(mean_abs_shap = mean(abs_shap)) %>%
        arrange(desc(mean_abs_shap))
    
    # Reorder features by importance
    shap_long$feature <- factor(shap_long$feature, 
                              levels = feature_importance$feature)
    
    # Create plot
    ggplot(shap_long, aes(x = feature, y = shap_value)) +
        geom_violin(fill = "lightblue", alpha = 0.5) +
        geom_boxplot(width = 0.1, fill = "white", alpha = 0.5) +
        coord_flip() +
        theme_minimal() +
        labs(
            title = "SHAP Values Distribution by Feature",
            x = "Features",
            y = "SHAP value"
        )
}

# Create feature importance plot based on SHAP values
plot_shap_importance <- function(shap_values) {
    # Calculate mean absolute SHAP values
    importance <- colMeans(abs(shap_values)) %>%
        sort(decreasing = TRUE)
    
    # Create data frame for plotting
    importance_df <- data.frame(
        feature = names(importance),
        importance = importance
    ) %>%
        arrange(desc(importance)) %>%
        head(30)  # Top 30 features
    
    # Create plot
    ggplot(importance_df, aes(x = reorder(feature, importance), y = importance)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        coord_flip() +
        theme_minimal() +
        labs(
            title = "Feature Importance Based on SHAP Values",
            x = "Features",
            y = "Mean |SHAP value|"
        )
}

# Create individual explanation plot
plot_individual_shap <- function(shap_values, observation_index = 1) {
    # Get SHAP values for single observation
    single_shap <- shap_values[observation_index,]
    
    # Create waterfall plot data
    waterfall_data <- data.frame(
        feature = names(single_shap),
        shap_value = as.numeric(single_shap)
    ) %>%
        arrange(desc(abs(shap_value))) %>%
        head(20)  # Top 20 features for clarity
    
    # Calculate cumulative values
    waterfall_data$cumulative <- cumsum(waterfall_data$shap_value)
    
    # Create plot
    ggplot(waterfall_data, aes(x = reorder(feature, abs(shap_value)))) +
        geom_bar(aes(y = shap_value, fill = shap_value > 0), 
                stat = "identity") +
        geom_point(aes(y = cumulative), color = "red") +
        geom_line(aes(y = cumulative, group = 1), color = "red") +
        coord_flip() +
        theme_minimal() +
        scale_fill_manual(values = c("red", "blue")) +
        labs(
            title = paste("Individual SHAP Explanation for Observation", observation_index),
            x = "Features",
            y = "SHAP value"
        ) +
        theme(legend.position = "none")
}

# Generate and save all plots
summary_plot <- plot_shap_summary(shap_values, train_features)
importance_plot <- plot_shap_importance(shap_values)
individual_plot <- plot_individual_shap(shap_values, 1)  # First observation

# Save plots
ggsave("shap_summary.png", summary_plot, width = 12, height = 10)
ggsave("shap_importance.png", importance_plot, width = 12, height = 8)
ggsave("shap_individual.png", individual_plot, width = 12, height = 8)

# Create feature importance table
shap_importance <- data.frame(
    Feature = colnames(shap_values),
    Mean_SHAP = colMeans(abs(shap_values))
) %>%
    arrange(desc(Mean_SHAP))

# Print top 20 most important features
print("Top 20 Most Important Features (SHAP):")
print(head(shap_importance, 20))

# Compare with original feature importance
print("\nComparison with Original Feature Importance:")
comparison <- data.frame(
    Feature = shap_importance$Feature,
    SHAP_Importance = shap_importance$Mean_SHAP,
    Original_Importance = importances$Overall[match(shap_importance$Feature, 
                                                  importances$Feature)]
) %>%
    arrange(desc(SHAP_Importance)) %>%
    head(20)

print(comparison)
