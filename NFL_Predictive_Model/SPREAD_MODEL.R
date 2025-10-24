par(mfrow = c(1, 1))
boxplot(data$result, main = 'Total points boxplot. 2017–2024 NFL data')
hist(data$result, 
     breaks = 20, 
     col = "lightblue", 
     border = "white",
     freq = FALSE,          # important: shows density, not counts
     main = "Distribution of total with normal curve",
     xlab = "Result")

spread_data <- data %>% filter(result < 40,
                               result > -35)

boxplot(spread_data$result, main = "Spread boxplot")
hist(spread_data$result, 
     breaks = 20, 
     col = "lightblue", 
     border = "white",
     freq = FALSE,          # important: shows density, not counts
     main = "Histogram of Spread",
     xlab = "Spread")


# Select variables to use in the model
vars_model <- spread_data %>%
  dplyr::select(result, week, season, weekday, stadium, home_rest, away_rest, 
                away_wins, away_losses, away_ties,
                home_wins, home_losses, home_ties,
                home_games, away_games,
                matches("avg"))

vars_model <- vars_model %>%
  mutate(
    weekday = as.factor(weekday),
    week = as.factor(week),
    stadium = as.factor(stadium)
  )

train <- vars_model %>% filter(season < 2024)
test  <- vars_model %>% filter(season == 2024)

# --- Baseline model ---------------------------------------------------------
spread_baseline_model <- lm(result ~ I(home_wins/home_games) + I(away_wins/away_games),
                            data = train)
summary(spread_baseline_model)

# --- 1️⃣ Cross-validation on training set ---
k <- length(unique(train$season))  # Leave-One-Season-Out
folds <- train$season
mae_cv <- c()

for(i in unique(folds)){
  train_fold <- train %>% filter(season != i)
  val_fold   <- train %>% filter(season == i)
  
  model_fold <- lm(result ~ I(home_wins/home_games) + I(away_wins/away_games),
                   data = train_fold)
  
  pred <- predict(model_fold, newdata = val_fold)
  mae_cv[i] <- mean(abs(val_fold$result - pred))
}

cat("📊 Mean MAE (CV on train):", mean(mae_cv, na.rm = TRUE), "\n")

# --- 2️⃣ Error on 2024 test set ---
pred_test <- predict(spread_baseline_model, newdata = test)
mae_test <- mean(abs(test$result - pred_test))
cat("📊 MAE on 2024 test set:", mae_test, "\n")


# --- Advanced model ---------------------------------------------------------
spread_model <- lm(result ~ 
                     home_avg_result + away_avg_result +
                     away_avg_spread_line + 
                     home_avg_epa_per_play + away_avg_epa_per_play,
                   data = train)
summary(spread_model)

# --- 1️⃣ Cross-validation on training set ---
k <- length(unique(train$season))  # Leave-One-Season-Out
folds <- train$season
mae_cv <- c()

for(i in unique(folds)){
  train_fold <- train %>% filter(season != i)
  val_fold   <- train %>% filter(season == i)
  
  model_fold <- lm(result ~ 
                     home_avg_result + away_avg_result +
                     away_avg_spread_line + 
                     home_avg_epa_per_play + away_avg_epa_per_play,
                   data = train_fold)
  
  pred <- predict(model_fold, newdata = val_fold)
  mae_cv[i] <- mean(abs(val_fold$result - pred))
}

cat("📊 Mean MAE (CV on train):", mean(mae_cv, na.rm = TRUE), "\n")

# --- 2️⃣ Error on 2024 test set ---
pred_test <- predict(spread_model, newdata = test)
mae_test <- mean(abs(test$result - pred_test))
cat("📊 MAE on 2024 test set:", mae_test, "\n")


# GLM --------------------------------------------------------------------------
# --- GLM ---
glm_model <- glm(result ~ home_avg_result + away_avg_result +
                   away_avg_spread_line + home_avg_epa_per_play + away_avg_epa_per_play,
                 data = train,
                 family = Gamma(link = 'log'))  # you can change to Gamma(link="log") if you want

# --- Leave-One-Season-Out CV on training set ---
folds <- unique(train$season)
mae_cv <- c()

for(f in folds){
  train_fold <- train %>% filter(season != f)
  val_fold   <- train %>% filter(season == f)
  
  model_fold <- glm(result ~ home_avg_result + away_avg_result +
                      away_avg_spread_line + home_avg_epa_per_play + away_avg_epa_per_play,
                    data = train_fold,
                    family = Gamma(link = 'log'))
  
  pred <- predict(model_fold, newdata = val_fold, type = "response")
  mae_cv <- c(mae_cv, mean(abs(val_fold$result - pred)))
}

cat("📊 Mean MAE (GLM CV on train):", mean(mae_cv), "\n")

# --- Error on 2024 test set ---
pred_test <- predict(glm_model, newdata = test, type = "response")
mae_test <- mean(abs(test$result - pred_test))
cat("📊 GLM MAE on 2024 test set:", mae_test, "\n")


# ELASTIC NET -------------------------------------------------------------------
# --- Prepare numeric matrices ---
x_train <- model.matrix(result ~ home_avg_result + away_avg_result +
                          away_avg_spread_line + home_avg_epa_per_play + away_avg_epa_per_play,
                        data = train)[, -1]  # removes intercept
y_train <- train$result

x_test <- model.matrix(result ~ home_avg_result + away_avg_result +
                         away_avg_spread_line + home_avg_epa_per_play + away_avg_epa_per_play,
                       data = test)[, -1]
y_test <- test$result

# --- CV glmnet (Lasso, MAE) ---
set.seed(123)
cvfit <- cv.glmnet(x_train, y_train,
                   alpha = 0.2,               
                   type.measure = "mae")

best_lambda <- cvfit$lambda.min
cat("📌 Optimal lambda (GLMNET):", best_lambda, "\n")

# --- Predictions on test set ---
pred_glmnet <- predict(cvfit, newx = x_test, s = "lambda.min")
mae_glmnet <- mean(abs(y_test - pred_glmnet))
cat("📊 MAE GLMNET on 2024 test set:", mae_glmnet, "\n")

# --- MAE cross-validation on training set (output from cv.glmnet) ---
mae_cv_glmnet <- mean(cvfit$cvm)
cat("📊 MAE GLMNET cross-validation on train:", mae_cv_glmnet, "\n")


# FINAL MODEL ---------------------------------------------------------
spread_model <- lm(result ~ 
                     home_avg_result + away_avg_result +
                     away_avg_spread_line + 
                     home_avg_epa_per_play + away_avg_epa_per_play,
                   data = data %>% filter(season < 2025))

summary(spread_model)

# --- 1️⃣ Filter 2025 data ---
data_2025 <- data %>% filter(season == 2025)

# --- 2️⃣ Predictions ---
pred_2025 <- predict(spread_model, newdata = data_2025)

# --- 3️⃣ Create dataframe with required information ---
predictions_spread_2025 <- data.frame(
  game_id   = data_2025$game_id,
  home_team = data_2025$home_team,
  away_team = data_2025$away_team,
  predicted_result = pred_2025
)


#----------------------
folds <- unique(train$season)
mae_cv <- c()
mae_by_week <- data.frame()

for(f in folds){
  train_fold <- train %>% filter(season != f)
  val_fold   <- train %>% filter(season == f)
  
  model_fold <- lm(result ~ 
                     home_avg_result + away_avg_result +
                     away_avg_spread_line + 
                     home_avg_epa_per_play + away_avg_epa_per_play,
                   data = train_fold)
  
  pred <- predict(model_fold, newdata = val_fold)
  
  # --- MAE by week ---
  mae_week_fold <- val_fold %>%
    mutate(pred = pred,
           abs_error = abs(result - pred)) %>%
    group_by(week) %>%
    summarise(MAE = mean(abs_error))
  
  mae_by_week <- bind_rows(mae_by_week, mae_week_fold)
  
  # --- Average MAE per fold ---
  mae_cv <- c(mae_cv, mean(abs(val_fold$result - pred)))
}

# --- Overall average MAE per week across all folds ---
mae_week_avg <- mae_by_week %>%
  group_by(week) %>%
  summarise(MAE_avg = mean(MAE))

cat("📊 Mean MAE Leave-One-Season-Out:", mean(mae_cv), "\n")
print(mae_week_avg)


###############################
prediction_2025 <- predictions_spread_2025 %>% 
  left_join(predictions__points_2025, by = c('game_id', 'home_team', 'away_team')) %>% 
  left_join(predictions_pass_2025, by = c('game_id', 'home_team', 'away_team'))

predictions <- read.csv('Predictions.csv')
prediction_2025 <- read.csv('PREDICTIONS_2025')

# Map of team names to their abbreviations
team_abbr <- c(
  "Arizona Cardinals" = "ARI",
  "Atlanta Falcons" = "ATL",
  "Baltimore Ravens" = "BAL",
  "Buffalo Bills" = "BUF",
  "Carolina Panthers" = "CAR",
  "Chicago Bears" = "CHI",
  "Cincinnati Bengals" = "CIN",
  "Cleveland Browns" = "CLE",
  "Dallas Cowboys" = "DAL",
  "Denver Broncos" = "DEN",
  "Detroit Lions" = "DET",
  "Green Bay Packers" = "GB",
  "Houston Texans" = "HOU",
  "Indianapolis Colts" = "IND",
  "Jacksonville Jaguars" = "JAX",
  "Kansas City Chiefs" = "KC",
  "Las Vegas Raiders" = "LV",
  "Los Angeles Chargers" = "LAC",
  "Los Angeles Rams" = "LA",
  "Miami Dolphins" = "MIA",
  "Minnesota Vikings" = "MIN",
  "New England Patriots" = "NE",
  "New Orleans Saints" = "NO",
  "New York Giants" = "NYG",
  "New York Jets" = "NYJ",
  "Philadelphia Eagles" = "PHI",
  "Pittsburgh Steelers" = "PIT",
  "San Francisco 49ers" = "SF",
  "Seattle Seahawks" = "SEA",
  "Tampa Bay Buccaneers" = "TB",
  "Tennessee Titans" = "TEN",
  "Washington Commanders" = "WAS"
)

# Remove any leading/trailing spaces
predictions$Home <- substr(predictions$Home, 2, nchar(predictions$Home))
predictions$Away <- substr(predictions$Away, 2, nchar(predictions$Away))

# Add columns with abbreviations
predictions$home_team <- team_abbr[predictions$Home]
predictions$away_team <- team_abbr[predictions$Away]

predictions$home_team[58] = 'MIA'

predictions <- predictions %>% 
  left_join(prediction_2025, by = c('home_team', 'away_team')) 

predictions$Spread <- round(predictions$predicted_result)
predictions$Total  <- round(predictions$predicted_total)
predictions$Pass   <- round(predictions$predicted_total_passing_yards)

final_predictions <- predictions %>% select(Week, Home, Away, Spread, Total, Pass)
write.csv(final_predictions, 'Predictions.csv', row.names = F)

