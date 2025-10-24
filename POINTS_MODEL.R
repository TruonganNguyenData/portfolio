boxplot(data$total)
hist(data$total, 
     breaks = 20, 
     col = "lightblue", 
     border = "white",
     freq = FALSE,          # important: shows density, not counts
     main = "Distribution of total with normal curve",
     xlab = "total")

data_2025 <- data %>% filter(season == 2025)

total_data <- data %>% filter(season < 2025, !is.na(total))

total_data <- data %>% filter(total > 9,
                              total < 105)

boxplot(total_data$total, main = "Points boxplot")
hist(total_data$total, 
     breaks = 20, 
     col = "lightblue", 
     border = "white",
     freq = FALSE,          # important: shows density, not counts
     main = "Histogram of Points",
     xlab = "Points")


# Select variables to use in the model
vars_model <- total_data %>%
  dplyr::select(total, week, season, weekday, stadium, home_rest, away_rest, 
                div_game,
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
total_baseline_model <- lm(total ~ home_avg_points_for + away_avg_points_for,
                           data = train)

# --- 1️⃣ Cross-validation on training set ---
k <- length(unique(train$season))  # Leave-One-Season-Out
folds <- train$season
mae_cv <- c()

for(i in unique(folds)){
  train_fold <- train %>% filter(season != i)
  val_fold   <- train %>% filter(season == i)
  
  model_fold <- lm(total ~ home_avg_points_for + away_avg_points_for,
                   data = train_fold)
  
  pred <- predict(model_fold, newdata = val_fold)
  mae_cv[i] <- mean(abs(val_fold$total - pred))
}

cat("📊 Mean MAE (CV on train):", mean(mae_cv, na.rm = TRUE), "\n")

# --- 2️⃣ Error on 2024 test set ---
pred_test <- predict(total_baseline_model, newdata = test)
mae_test <- mean(abs(test$total - pred_test))

cat("📊 MAE on 2024 test set:", mae_test, "\n")

summary(total_baseline_model)

# --- Advanced model ---------------------------------------------------------
total_model <- lm(total ~ 
                    I(home_avg_points_for^2) + I(away_avg_points_for^2)
                  #home_avg_fg_made + away_avg_fg_made 
                  ,
                  data = train)

summary(total_model)
par(mfrow = c(2, 2))
plot(total_model)

# --- 1️⃣ Cross-validation on training set ---
k <- length(unique(train$season))  # Leave-One-Season-Out
folds <- train$season
mae_cv <- c()

for(i in unique(folds)){
  train_fold <- train %>% filter(season != i)
  val_fold   <- train %>% filter(season == i)
  
  model_fold <- lm(total ~ 
                     I(home_avg_points_for^2) + I(away_avg_points_for^2)
                   #home_avg_fg_made + away_avg_fg_made ,
                   ,
                   data = train_fold)
  
  pred <- predict(model_fold, newdata = val_fold)
  mae_cv[i] <- mean(abs(val_fold$total - pred))
}

cat("📊 Mean MAE (CV on train):", mean(mae_cv, na.rm = TRUE), "\n")

# --- 2️⃣ Error on 2024 test set ---
pred_test <- predict(total_model, newdata = test)
mae_test <- mean(abs(test$total - pred_test))

cat("📊 MAE on 2024 test set:", mae_test, "\n")



# GLM --------------------------------------------------------------------------
# --- GLM ---
glm_model <- glm(total ~ I(home_avg_points_for^2) + I(away_avg_points_for^2)+ 
                   home_avg_fg_made + away_avg_fg_made,
                 data = train,
                 family = Gamma(link = 'log'))  # you can change to Gamma(link="log") if you want

# --- Leave-One-Season-Out CV on training set ---
folds <- unique(train$season)
mae_cv <- c()

for(f in folds){
  train_fold <- train %>% filter(season != f)
  val_fold   <- train %>% filter(season == f)
  
  model_fold <- glm(total ~ home_avg_total + away_avg_total +
                      away_avg_total_line + home_avg_epa_per_play + away_avg_epa_per_play,
                    data = train_fold,
                    family = Gamma(link = 'log'))
  
  pred <- predict(model_fold, newdata = val_fold, type = "response")
  mae_cv <- c(mae_cv, mean(abs(val_fold$total - pred)))
}

cat("📊 Mean MAE (GLM CV on train):", mean(mae_cv), "\n")

# --- Error on 2024 test set ---
pred_test <- predict(glm_model, newdata = test, type = "response")
mae_test <- mean(abs(test$total - pred_test))
cat("📊 GLM MAE on 2024 test set:", mae_test, "\n")


# ELASTIC NET -------------------------------------------------------------------
# --- Prepare numeric matrices ---
x_train <- model.matrix(total ~ I(home_avg_points_for^2) + I(away_avg_points_for^2)+ 
                          home_avg_fg_made + away_avg_fg_made, 
                        data = train)[, -1]  # removes intercept
y_train <- train$total

x_test <- model.matrix(total ~ I(home_avg_points_for^2) + I(away_avg_points_for^2)+ 
                         home_avg_fg_made + away_avg_fg_made ,
                       data = test)[, -1]
y_test <- test$total


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

# --- 6️⃣ Final fit on full training set with optimal alpha ---
final_fit <- cv.glmnet(x_train, y_train, alpha = best_alpha, type.measure = "mae")
best_lambda <- final_fit$lambda.min
cat("📌 Final optimal lambda:", best_lambda, "\n")

# --- 7️⃣ MAE on test (2025) ---
pred_test <- predict(final_fit, newx = x_test, s = "lambda.min")
mae_test <- mean(abs(y_test - pred_test))
cat("📊 MAE on 2025 test:", round(mae_test, 3), "\n")

# FINAL MODEL 
total_model <- lm(total ~ 
                    I(home_avg_points_for^2) + I(away_avg_points_for^2),
                  data = data %>% filter(season < 2025))

summary(total_model)
# --- 1️⃣ Filter 2025 data ---
data_2025 <- data %>% filter(season == 2025)


# --- 5️⃣ Evaluation on test set ---
pred_test <- predict(total_model, newdata = data_2025)


# --- 5️⃣ Create dataframe with predictions ---
predictions__points_2025 <- data.frame(
  game_id   = data_2025$game_id,
  home_team = data_2025$home_team,
  away_team = data_2025$away_team,
  predicted_total = as.numeric(pred_test)
)


# --- 2️⃣ Predictions ---
x_train <- model.matrix(total ~ I(home_avg_points_for^2) + I(away_avg_points_for^2)+ 
                          home_avg_fg_made + away_avg_fg_made, 
                        data = total_data)[, -1]  # removes intercept
y_train <- total_data$total

x_test <- model.matrix( ~ I(home_avg_points_for^2) + I(away_avg_points_for^2)+ 
                          home_avg_fg_made + away_avg_fg_made ,
                        data = data_2025)[, -1]


cvfit_final <- cv.glmnet(x_train, y_train, alpha = best_alpha, type.measure = "mae")
best_lambda <- cvfit_final$lambda.min
cat("📌 Optimal lambda:", best_lambda, "\n")

# --- 5️⃣ Evaluation on test set ---
pred_test <- predict(cvfit_final, newx = x_test, s = "lambda.min")


# --- 5️⃣ Create dataframe with predictions ---
predictions_2025 <- data.frame(
  game_id   = data_2025$game_id,
  home_team = data_2025$home_team,
  away_team = data_2025$away_team,
  predicted_total = as.numeric(pred_test)
)



pred_2025 <- predict(total_model, newdata = data_2025)

# --- 3️⃣ Create dataframe with required information ---
predictions_2025 <- data.frame(
  game_id   = data_2025$game_id,
  home_team = data_2025$home_team,
  away_team = data_2025$away_team,
  predicted_total = pred_2025
)
