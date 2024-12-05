install.packages("pacman")

require("pacman")

# Use pacman to load add-on packages as desired
pacman::p_load(pacman, rio, tidyverse, caret, Cubist, Metrics, ggplot2) 

# CSV
getwd()
train_features_file = "/Users/hoanguyen/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Recovery rate forecasting/Data/final/train_features2.xlsx"
train_labels_file = "/Users/hoanguyen/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Recovery rate forecasting/Data/final/train_labels2.xlsx"
test_features_file = "/Users/hoanguyen/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Recovery rate forecasting/Data/final/test_features2.xlsx"
test_labels_file = "/Users/hoanguyen/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Recovery rate forecasting/Data/final/test_labels2.xlsx"

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


###### FEATURE IMPORTANCES
importances = caret::varImp(model_rules)

# Add feature names as a new column
importances$Feature = rownames(importances)

# Sort by importance in descending order and select top 50
top_features = importances[order(-importances$Overall), ][1:50, ]

# Plot using ggplot2
ggplot(top_features, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_col(fill = "steelblue", width = 0.5) +
  coord_flip() +  # Flip the axes to make the plot horizontal
  labs(x = "Feature", y = "Importance", title = "Top 50 Feature Importances") +
  theme_minimal()  # Clean theme



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
