rsq = function (x, y) cor(x, y) ^ 2

# Readin Data into DataFrame
sem = read.csv('C:\\Users\\jim19\\Desktop\\Statistics_Project\\ML Models\\train.csv')

# Average RMSE in 25 iters
TrainRMSE_list = c()
TrainRSQ_list  = c()
TestRMSE_list  = c()
TestRSQ_list   = c()
for (iter in 1:25) {
  # Split into Traing & Test
  smp_size = floor(0.67 * nrow(sem))
  train_ind = sample(seq_len(nrow(sem)), size = smp_size)
  
  sem_train = sem[train_ind, ]
  sem_test = sem[-train_ind, ]
  
  # Apply Naive Bayes
  sem_ml = lm(sem_train$critical_temp ~ ., data=sem_train)
  
  # Train Error
  train_pred = predict.lm(sem_ml, sem_train)
  train_gt = sem_train$critical_temp
  TrainRMSE = sqrt( sum( (train_pred - train_gt)^2 ) / length(train_pred) )
  TrainRSQ  = rsq(train_pred, train_gt)
  
  # Test Error
  test_pred = predict.lm(sem_ml, sem_test)
  test_gt = sem_test$critical_temp
  TestRMSE = sqrt( sum( (test_pred - test_gt)^2 ) / length(test_pred) )
  TestRSQ  = rsq(test_pred, test_gt)
  
  # Collect Results
  TrainRMSE_list = c(TrainRMSE_list, TrainRMSE)
  TrainRSQ_list = c(TrainRSQ_list, TrainRSQ)
  TestRMSE_list = c(TestRMSE_list, TestRMSE)
  TestRSQ_list = c(TestRSQ_list, TestRSQ)
}

# Runtime 
ptm = proc.time()
sem_ml = lm(sem_train$critical_temp ~ ., data=sem_train)
proc.time() - ptm

# Average RMSE and RSQ
AveTrainRMSE = mean(TrainRMSE_list)
AveTrainRSQ  = mean(TrainRSQ_list)
AveTestRMSE  = mean(TestRMSE_list)
AveTestRSQ   = mean(TestRSQ_list)

AveTrainRMSE
AveTrainRSQ
AveTestRMSE
AveTestRSQ

# Plot TrueTc - PredTc for test batch(last iter)
plot(test_gt, test_pred)
lines(seq(1,150),seq(1,150),col="red")
