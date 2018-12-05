# Readin Data into DataFrame
sem = read.csv('C:\\Users\\jim19\\Desktop\\Statistics_Project\\train.csv')

# Split into Traing & Test
smp_size <- floor(0.75 * nrow(sem))
train_ind <- sample(seq_len(nrow(sem)), size = smp_size)

sem_train = sem[train_ind, ]
sem_test = sem[-train_ind, ]

# Apply Naive Bayes
sem_ml = lm(sem$critical_temp ~ ., data=sem_train)

# Train Error
train_pred = predict.lm(sem_ml, sem_train)
gt = sem_train$critical_temp

TrainRmse = sqrt( sum( (train_pred - gt)^2 ) / length(train_pred) )

# Test Error
test_pred = predict.lm(sem_ml, sem_test)
gt = sem_test$critical_temp

TestRmse = sqrt( sum( (test_pred - gt)^2 ) / length(test_pred) )

