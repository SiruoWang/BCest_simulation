library(dplyr)
library(caret)
library(parallel)

## create_testIndex(num) randomly samples data into 10 groups
create_testIndex <- function(num){
  Index_list <- list()
  group_idx <- sample(cut(seq(1,num),breaks=10,label=FALSE))
  for (i in 1:10){
    Index_list[[i]] <- which(group_idx == i)
  }

  return(Index_list)
}

simulation <- function(sample_size, num_of_cov, beta_list, Index_list, use_seed=12345){

  set.seed(use_seed)

  ## simulate x1 in training, testing, validation
  covariate_list <- list()

  covariate_list[[1]] <- rep(1, sample_size)
  for (i in c(2:(num_of_cov+1))){
    covariate_list[[i]] <- rnorm(sample_size, mean = 0, sd = 1) ## x1
  }
  ## simulate errors
  e <- rnorm(sample_size, mean = 0, sd = 1)

  ## partition simulated covariate matrix into testing, training, validation in propotion 6:2:2
  covariate_training <- list()
  covariate_testing <- list()
  covariate_validation <- list()
  for (i in 1:num_of_cov){
    covariate_training[[i]] <- covariate_list[[i+1]][unlist(Index_list[1:6])]
    covariate_testing[[i]] <- covariate_list[[i+1]][unlist(Index_list[7:8])]
    #covariate_list[[i+1]][unlist(Index_list[9:10])] <- covariate_list[[i+1]][unlist(Index_list[9:10])] + rnorm(sample_size/5, mean = 1, sd = 2)
    covariate_validation[[i]] <- covariate_list[[i+1]][unlist(Index_list[9:10])]
  }

  y <- Map('*', beta_list, covariate_list) %>% Reduce('+',.) + e
  #y <- Map('*', beta_list[1:(length(beta_list) - 1)], covariate_list[1:(length(beta_list) - 1)]) %>%
  #     Reduce('+',.) + beta_list[[length(beta_list)]] * covariate_list[[length(beta_list)]]^2 + e


  ## partition simulated true Y into testing, training, validation in propotion 6:2:2
  y_training <- y[unlist(Index_list[1:6])]
  y_testing <- y[unlist(Index_list[7:8])]
  y_validation <- y[unlist(Index_list[9:10])]

  ## fit a machine learning model (random forest -- 'method = ranger') using traing data Y ~ X, and get predictions for testing data sets
  training_data <- data.frame(Y = y_training, X = matrix(unlist(covariate_training), nrow = num_of_cov, byrow = TRUE) %>% t())
  # fitControl <- trainControl(## 5-fold CV
  #                          method = "repeatedcv",
  #                          number = 5,
  #                          repeats = 5)
  model <- train(Y ~., data = training_data,
             method = 'rf',
             importance = TRUE)

   # model <- train(Y ~., data = training_data,
   #           method='ranger',
   #           trControl = fitControl,
   #           importance="permutation")

   # model <- train(Y ~., data = training_data,
   #           method = 'bstTree',
   #           preProc=c('center','scale'))

   #model <- lm(Y ~., data = training_data)
  ## calculate expected y in the testing dataset using model (random forrest)
  testing_covariates <- data.frame(X = matrix(unlist(covariate_testing), nrow = num_of_cov, by = TRUE) %>% t())
  y_hat_testing <- predict(model, testing_covariates)

  validation_covariates <- data.frame(X = matrix(unlist(covariate_validation), nrow = num_of_cov, by = TRUE) %>% t())
  y_hat_validation <- predict(model, validation_covariates)

#########################################
## plot y_testing and y_hat_testing in 2D plot (explore joint distribution of y_testing and y_hat_testing)
## joint distribution of y_testing and y_hat_testing is approximately multivariate(bivariate) normal

# colors <- densCols(y_testing, y_hat_testing, bandwidth = 2)
# data <- data.frame(y_testing, y_hat_testing)
# p_yhat_y <- ggplot(data, aes(x = y_testing, y = y_hat_testing)) +
#             geom_point(color = colors, size = 2) +
#             geom_abline(linetype="dashed",color="red",size=1) +
#             labs(y="Predicted Y in testing set (LM)", x = "True Y in testing set")
# print(p_yhat_y)
# #
# colors <- densCols(covariate_validation[[1]], y_validation, bandwidth = 2)
# data <- data.frame(covariate_validation[[1]], y_validation)
# p_yhat_y <- ggplot(data, aes(x = covariate_validation[[1]], y = y_validation)) +
#             geom_point(color = colors, size = 2) +
#             geom_abline(linetype="dashed",color="red",size=1) +
#             labs(y="y val", x = "x val")
# print(p_yhat_y)

#########################################
  X_val_matrix <- cbind(rep(1,nrow(validation_covariates)),validation_covariates) %>% as.matrix()
  X_test_matrix <- cbind(rep(1,nrow(testing_covariates)),testing_covariates) %>% as.matrix()


  beta_hat_est <- solve(t(X_val_matrix) %*% X_val_matrix) %*% t(X_val_matrix) %*% y_hat_validation

  Y_test_mu <- mean(y_testing)
  Y_hat_test_mu <- mean(y_hat_testing)

  gamma1_mle <- sum((y_testing - Y_test_mu) * (y_hat_testing - Y_hat_test_mu)) /
                sum((y_testing - Y_test_mu)^2)
  gamma0_mle <- Y_hat_test_mu - gamma1_mle * Y_test_mu
  beta_hat_test <- solve(t(X_test_matrix) %*% X_test_matrix) %*% t(X_test_matrix) %*% y_testing
  beta_hat_test_yhat <- solve(t(X_test_matrix) %*% X_test_matrix) %*% t(X_test_matrix) %*% y_hat_testing
#########################################
  ## plot y_hat_validation and est_hat_validation in 2D plot
  # est_hat_validation <- gamma0_mle + gamma1_mle * y_validation#X_val_matrix %*% beta_hat_test
  # colors <- densCols(y_hat_validation, est_hat_validation, bandwidth = 2)
  # data <- data.frame(y_hat_validation, est_hat_validation)
  # p_yhat_y <- ggplot(data, aes(x = y_hat_validation, y = est_hat_validation)) +
  #             geom_point(color = colors, size = 2) +
  #             geom_abline(linetype="dashed",color="red",size=1) +
  #             labs(y="estimated Y_hat in validation set (MLE)", x = "Y_hat in validation set")
  # print(p_yhat_y)
  # # # # #
  # est_validation <- X_val_matrix %*% beta_hat_test
  # colors <- densCols(y_validation, est_validation, bandwidth = 2)
  # data <- data.frame(y_validation, est_validation)
  # p_y <- ggplot(data, aes(x = y_validation, y = est_validation)) +
  #            geom_point(color = colors, size = 2) +
  #            ggtitle("beta1=3") +
  #            geom_abline(linetype="dashed",color="red",size=1) +
  #            labs(y="estimated Y in validation set (MLE)", x = "Y in validation set")
  # print(p_y)
  #########################################

  # Bias1 <- (solve(t(X_val_matrix) %*% X_val_matrix) %*% t(X_val_matrix)) %*%
  #        (gamma0_mle + gamma1_mle * X_val_matrix %*% beta_hat_test) - beta_hat_test


  Bias2 <-  beta_hat_test_yhat -  beta_hat_test
  beta_hat_BCest <- beta_hat_est - Bias2

  sigma_Y_hat_test <- (sum((y_hat_testing - gamma0_mle - gamma1_mle * y_testing)^2) / (length(y_testing)-1))
  sigma_Y_test <- sum((y_testing - X_test_matrix %*% beta_hat_test)^2) / (length(y_testing)-1)


  var_BCest <- solve(t(X_val_matrix) %*% X_val_matrix) * (sigma_Y_hat_test + gamma1_mle^2 * sigma_Y_test)

  g_xval <- Map('*', beta_list, c(rep(1,length(covariate_validation[[1]])) %>% list(),covariate_validation)) %>% Reduce('+',.)
  #g_xval <- Map('*', beta_list[1:(length(beta_list) - 1)], c(rep(1,length(covariate_validation[[1]])) %>% list(), covariate_list[1:(length(beta_list) - 1)])) %>%
       #Reduce('+',.) + beta_list[[length(beta_list)]] * covariate_list[[length(beta_list)-1]]^2

  beta_true <- solve(t(X_val_matrix) %*% X_val_matrix) %*% t(X_val_matrix) %*% g_xval

  beta_val <- solve(t(X_val_matrix) %*% X_val_matrix) %*% t(X_val_matrix) %*% y_validation


  #### t statistics for testing beta = beta_true
  ## lm using known Y_val
  t_b_val <- (beta_val - beta_true) / sqrt(diag(solve(t(X_val_matrix) %*% X_val_matrix) * sum((y_validation - X_val_matrix %*% beta_val)^2)/(length(y_validation - 1))))
  ## predicted Y_hat_val with lm standard error
  t_b_est_lmse <- (beta_hat_est - beta_true) / sqrt(diag(solve(t(X_val_matrix) %*% X_val_matrix) * sum((y_hat_validation - X_val_matrix %*% beta_hat_est)^2)/(length(y_validation - 1))))
  ## bias corrected beta with lm standard error
  t_b_BCest_lmse <- (beta_hat_est - Bias2 - beta_true) / sqrt(diag(solve(t(X_val_matrix) %*% X_val_matrix) * sum((y_hat_validation - X_val_matrix %*% beta_hat_est)^2)/(length(y_validation - 1))))
  ## predicted Y_hat_val with bias corrected standard error
  t_b_est_BCse <- (beta_hat_est - beta_true) / sqrt(diag(var_BCest))
  ## bias corrected beta with bias corrected standard error
  t_b_BCest_BCse <- (beta_hat_est - Bias2 - beta_true) / sqrt(diag(var_BCest))

  #### t statistics for testing beta = 0 (in the summary table when fitting a linear model)
  t_0_val <- beta_val / sqrt(diag(solve(t(X_val_matrix) %*% X_val_matrix) * sum((y_validation - X_val_matrix %*% beta_val)^2)/(length(y_validation - 1))))
  t_0_est_lmse <- beta_hat_est / sqrt(diag(solve(t(X_val_matrix) %*% X_val_matrix) * sum((y_hat_validation - X_val_matrix %*% beta_hat_est)^2)/(length(y_validation - 1))))
  t_0_BCest_lmse <- (beta_hat_est - Bias2) / sqrt(diag(solve(t(X_val_matrix) %*% X_val_matrix) * sum((y_hat_validation - X_val_matrix %*% beta_hat_est)^2)/(length(y_validation - 1))))
  t_0_est_BCse <- beta_hat_est / sqrt(diag(var_BCest))
  t_0_BCest_BCse <- (beta_hat_est - Bias2) / sqrt(diag(var_BCest))

  #######################
  #testing <- data.frame(y = y_testing, y_pred = y_hat_testing, testing_covariates)
  #validation <- data.frame(y = y_validation, y_pred = y_hat_validation, validation_covariates)
  #write.table(testing, file = "testing.csv", quote = TRUE, sep = " ", col.names = TRUE)
  #write.table(validation, file = "validation.csv", quote = TRUE, sep = " ", col.names = TRUE)
  ######################

  rmse_beta_hat_est <- sqrt(mean((beta_hat_est[-1] - beta_true[-1])^2))
  rmse_beta_hat_BCest <- sqrt(mean((beta_hat_BCest[-1] - beta_true[-1])^2))


  result <- list(Bias2, beta_hat_est, beta_hat_test, beta_val, beta_hat_test_yhat, beta_true,
                rmse_beta_hat_est, rmse_beta_hat_BCest,
                t_b_val, t_b_est_lmse, t_b_BCest_lmse, t_b_est_BCse, t_b_BCest_BCse,
                t_0_val, t_0_est_lmse, t_0_BCest_lmse, t_0_est_BCse, t_0_BCest_BCse)
  return(result)
}

num_of_cov = 1
sample_size = 2000

for (beta2 in seq(1,40,1)){

  beta_list <- list(1,beta2)

  set.seed(2017)
  Index_list <- create_testIndex(sample_size)

  result <- mclapply(c(1:200), simulation, sample_size = sample_size, num_of_cov = num_of_cov, beta_list = beta_list, Index_list = Index_list,mc.cores = 20)
  save(result, file = paste0("result_beta",beta2,".rda"))

}

q(save = "no")
