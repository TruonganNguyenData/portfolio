data_2024 <- data %>% filter(season < 2025)

boxplot(data_2024$total_passing_yards)
hist(data$total_passing_yards, 
     breaks = 20, 
     col = "lightblue", 
     border = "white",
     freq = FALSE,          # important: shows density, not counts
     main = "Distribution of total with normal curve",
     xlab = "total_passing_yards")

total_passing_yards_data <- data %>% filter(total_passing_yards < 760)

boxplot(total_passing_yards_data$total_passing_yards, main = "total_passing_yards boxplot")
hist(total_passing_yards_data$total_passing_yards, 
     breaks = 20, 
     col = "lightblue", 
     border = "white",
     freq = FALSE,          # important: shows density, not counts
     main = "Histogram of total_passing_yards",
     xlab = "total_passing_yards")


# Select variables to use in the model
vars_model <- total_passing_yards_data %>%
  dplyr::select(total_passing_yards, week, season, weekday, stadium, home_rest, away_rest, 
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
total_passing_yards_baseline_model <- lm(total_passing_yards ~ home_avg_passing_yards + away_avg_passing_yards,
                                         data = train)

total_passing_yards_baseline_model <- lm(total_passing_yards ~ home_avg_passing_yards + away_avg_passing_yards,
                                         data = train)

# --- 1️⃣ Cross-validation on the training set ---
k <- length(unique(train$season))  # Leave-One-Season-Out
folds <- train$season
mae_cv <- c()

for(i in unique(folds)){
  train_fold <- train %>% filter(season != i)
  val_fold   <- train %>% filter(season == i)
  
  model_fold <- lm(total_passing_yards ~ home_avg_passing_yards + away_avg_passing_yards,
                   data = train)
  
  pred <- predict(model_fold, newdata = val_fold)
  mae_cv[i] <- mean(abs(val_fold$total_passing_yards - pred))
}

cat("📊 Mean MAE (CV on train):", mean(mae_cv, na.rm = TRUE), "\n")

# --- 2️⃣ Error on 2024 test set ---
pred_test <- predict(total_passing_yards_baseline_model, newdata = test)
mae_test <- mean(abs(test$total_passing_yards - pred_test))

cat("📊 MAE on 2024 test set:", mae_test, "\n")


summary(total_passing_yards_baseline_model)
plot(total_passing_yards_baseline_model)

# --- Advanced model ---------------------------------------------------------
total_passing_yards_model <- lm(total_passing_yards ~ 
                                  home_avg_passing_epa +
                                  away_avg_passing_cpoe + home_avg_passing_cpoe ,
                                data = train)
summary(total_passing_yards_model)
plot(total_passing_yards_model)

# --- 1️⃣ Cross-validation on the training set ---
k <- length(unique(train$season))  # Leave-One-Season-Out
folds <- train$season
mae_cv <- c()

for(i in unique(folds)){
  train_fold <- train %>% filter(season != i)
  val_fold   <- train %>% filter(season == i)
  
  model_fold <- lm(total_passing_yards ~ 
                     home_avg_passing_epa +
                     away_avg_passing_cpoe + home_avg_passing_cpoe ,
                   data = train_fold)
  
  pred <- predict(model_fold, newdata = val_fold)
  mae_cv[i] <- mean(abs(val_fold$total_passing_yards - pred))
}

cat("📊 Mean MAE (CV on train):", mean(mae_cv, na.rm = TRUE), "\n")

# --- 2️⃣ Error on 2024 test set ---
pred_test <- predict(total_passing_yards_model, newdata = test)
mae_test <- mean(abs(test$total_passing_yards - pred_test))

cat("📊 MAE on 2024 test set:", mae_test, "\n")



# GLM --------------------------------------------------------------------------
# --- GLM ---
glm_model <- glm(total_passing_yards ~ 
                   home_avg_passing_epa +
                   away_avg_passing_cpoe + home_avg_passing_cpoe,
                 data = train,
                 family = Gamma(link = 'log'))  # you can change to Gamma(link="log") if you want

# --- Leave-One-Season-Out CV on the training set ---
folds <- unique(train$season)
mae_cv <- c()

for(f in folds){
  train_fold <- train %>% filter(season != f)
  val_fold   <- train %>% filter(season == f)
  
  model_fold <- glm(total_passing_yards ~ 
                      home_avg_passing_epa +
                      away_avg_passing_cpoe + home_avg_passing_cpoe ,
                    data = train_fold,
                    family = Gamma(link = 'log'))
  
  pred <- predict(model_fold, newdata = val_fold, type = "response")
  mae_cv <- c(mae_cv, mean(abs(val_fold$total_passing_yards - pred)))
}

cat("📊 Mean MAE (GLM CV on train):", mean(mae_cv), "\n")

# --- Error on 2024 test set ---
pred_test <- predict(glm_model, newdata = test, type = "response")
mae_test <- mean(abs(test$total_passing_yards - pred_test))
cat("📊 GLM MAE on 2024 test set:", mae_test, "\n")


# ELASTIC NET -------------------------------------------------------------------
# --- Prepare numeric matrices ---
x_train <- model.matrix(total_passing_yards ~ 
                          home_avg_passing_yards +
                          away_avg_passing_yards ,
                        data = train)[, -1]  # removes intercept
y_train <- train$total_passing_yards

x_test <- model.matrix(total_passing_yards ~ 
                         home_avg_passing_yards +
                         away_avg_passing_yards ,
                       data = test)[, -1]
y_test <- test$total_passing_yards

# --- CV glmnet (Lasso, MAE) ---
alpha_grid <- seq(0, 1, by = 0.1)

# --- 4️⃣ Leave-One-Season-Out Cross-validation for each alpha ---
seasons <- unique(train$season)
cv_results <- data.frame(alpha = numeric(), mae_cv = numeric())

for (a in alpha_grid) {
  mae_cv_alpha <- c()
  
  for (s in seasons) {
    # training set without season s
    x_tr <- x_train[train$season != s, ]
    y_tr <- y_train[train$season != s]
    
    # validation = season s
    x_val <- x_train[train$season == s, ]
    y_val <- y_train[train$season == s]
    
    # model with specific alpha
    cvfit <- cv.glmnet(x_tr, y_tr, alpha = a, type.measure = "mae")
    
    # out-of-season predictions
    pred_val <- predict(cvfit, newx = x_val, s = "lambda.min")
    mae_cv_alpha <- c(mae_cv_alpha, mean(abs(y_val - pred_val)))
  }
  
  mae_mean_alpha <- mean(mae_cv_alpha)
  cv_results <- rbind(cv_results, data.frame(alpha = a, mae_cv = mae_mean_alpha))
  
  cat("✅ Alpha =", a, "| Mean MAE LOSO:", round(mae_mean_alpha, 3), "\n")
}

# --- 5️⃣ Optimal alpha ---
best_alpha <- cv_results$alpha[which.min(cv_results$mae_cv)]
cat("\n🏆 Optimal alpha:", best_alpha, "\n")

# --- 6️⃣ Final fit on the entire training set with optimal alpha ---
final_fit <- cv.glmnet(x_train, y_train, alpha = best_alpha, type.measure = "mae")
best_lambda <- final_fit$lambda.min
cat("📌 Final optimal lambda:", best_lambda, "\n")

# FINAL MODEL 
total_passing_yards_model <- lm(total_passing_yards ~ 
                                  home_avg_passing_yards + away_avg_passing_yards,
                                data = data %>% filter(season < 2025))

summary(total_passing_yards_model)

# --- 1️⃣ Filter 2025 data ---
data_2025 <- data %>% filter(season == 2025)

# --- 2️⃣ Predictions ---
pred_2025 <- predict(total_passing_yards_model, newdata = data_2025)

# --- 3️⃣ Create dataframe with the required information ---
predictions_pass_2025 <- data.frame(
  game_id   = data_2025$game_id,
  home_team = data_2025$home_team,
  away_team = data_2025$away_team,
  predicted_total_passing_yards = pred_2025
)


#----------------------
folds <- unique(train$season)
mae_cv <- c()
mae_by_week <- data.frame()

for(f in folds){
  train_fold <- train %>% filter(season != f)
  val_fold   <- train %>% filter(season == f)
  
  model_fold <- lm(total_passing_yards ~ 
                     home_avg_total_passing_yards + away_avg_total_passing_yards +
                     away_avg_total_passing_yards_line + 
                     home_avg_epa_per_play + away_avg_epa_per_play,
                   data = train_fold)
  
  pred <- predict(model_fold, newdata = val_fold)
  
  # --- MAE by week ---
  mae_week_fold <- val_fold %>%
    mutate(pred = pred,
           abs_error = abs(total_passing_yards - pred)) %>%
    group_by(week) %>%
    summarise(MAE = mean(abs_error))
  
  mae_by_week <- bind_rows(mae_by_week, mae_week_fold)
  
  # --- Average MAE per fold ---
  mae_cv <- c(mae_cv, mean(abs(val_fold$total_passing_yards - pred)))
}

# --- Overall average MAE per week across all folds ---
mae_week_avg <- mae_by_week %>%
  group_by(week) %>%
  summarise(MAE_avg = mean(MAE))

cat("📊 Mean MAE Leave-One-Season-Out:", mean(mae_cv), "\n")
print(mae_week_avg)
