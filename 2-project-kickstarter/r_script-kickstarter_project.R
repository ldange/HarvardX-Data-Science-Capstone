##########################################################
# Create kickstarter train set, validation set (final hold-out test set, similar to the edx set for the Movielens Project)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(stringr)

# Kickstarter's Projects Database:
# https://www.kaggle.com/kemical/kickstarter-projects?select=ks-projects-201612.csv
# https://www.kaggle.com/kemical/kickstarter-projects/download

# Data can be viewed/downloaded from my github repository
# https://github.com/ldange/HarvardX-Data-Science-Capstone/blob/master/ks-projects-201801.csv

# Creating the dataframe from csv file
kickstarter <- read.csv("https://raw.githubusercontent.com/ldange/HarvardX-Data-Science-Capstone/master/ks-projects-201801.csv")

# Removing unecessary column
drops <- c("deadline","usd.pledged","usd_pledged_real","usd_goal_real")
kickstarter <- kickstarter[ , !(names(kickstarter) %in% drops)]

# Renaming and Reordering column
kickstarter <- kickstarter %>% 
  rename(
    sub_category = category,
    status = state,
    projectId = ID
  )
kickstarter <- kickstarter[ , c(1,2,4,3,11,5,7,10,8,6,9)]

# Creating subset datafram without the status "live, suspended & undefined"
kickstarter <- subset(kickstarter, kickstarter$status %in% c("canceled","failed","successful") )

# Validation set will be 10% of Kickstarter Project data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index_kickstarter <- createDataPartition(y = kickstarter$status, times = 1, p = 0.1, list = FALSE)
train_kickstarter <- kickstarter[-test_index_kickstarter,]
validation_kickstarter <- kickstarter[test_index_kickstarter,]

#Machine Learning Projects

#Compute mu & predict unknown funding based on mu
mu <- mean(train_kickstarter$pledged)
average_funding_project <- RMSE(train_kickstarter$pledged, mu)
average_funding_project

#compute the movie bias b_i
b_i <- train_kickstarter %>%
  group_by(main_category) %>%
  summarize(b_i = mean(pledged - mu))

#Predicting the funding with mu and the main_category bias term.
predicted_funding_1 <- validation_kickstarter %>% 
  left_join(b_i, by='main_category') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
average_funding_project_w_category_biais <- RMSE(predicted_funding_1, validation_kickstarter$pledged)
average_funding_project_w_category_biais

# Compute the bias term, b_u
b_u <- train_kickstarter %>% 
  left_join(b_i, by='main_category') %>%
  group_by(sub_category) %>%
  summarize(b_u = mean(pledged - mu - b_i))

#Predicting the funding with mu, the main_category and sub_category bias term.
predicted_funding_2 <- validation_kickstarter %>% 
  left_join(b_i, by='main_category') %>%
  left_join(b_u, by='sub_category') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

avg_movie_rating_w_subcategory_bias <- RMSE(predicted_funding_2, validation_kickstarter$pledged)
avg_movie_rating_w_subcategory_bias

# Compute the bias term, b_u
b_u <- train_kickstarter %>% 
  left_join(b_i, by='main_category') %>%
  group_by(country) %>%
  summarize(b_u = mean(pledged - mu - b_i))

#Predicting the funding with mu, the main_category and country bias term.
predicted_funding_3 <- validation_kickstarter %>% 
  left_join(b_i, by='main_category') %>%
  left_join(b_u, by='country') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

avg_movie_rating_w_country_bias <- RMSE(predicted_funding_3, validation_kickstarter$pledged)
avg_movie_rating_w_country_bias

#Failed attempt of the regularization
lambdas <- seq(from=25000, to=100000, by=2500)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_kickstarter$pledged)
  
  b_i <- train_kickstarter %>%
    group_by(main_category) %>%
    summarize(b_i = sum(pledged - mu)/(n()+l))
  
  b_u <- train_kickstarter %>% 
    left_join(b_i, by='main_category') %>%
    group_by(country) %>%
    summarize(b_u = sum(pledged - b_i - mu)/(n()+l))
  
  predicted_funding_3 <- validation_kickstarter %>% 
    left_join(b_i, by = "main_category") %>%
    left_join(b_u, by = "country") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_funding_3, validation_kickstarter$rating))
})

qplot(lambdas, rmses)  
lambdas[which.min(rmses)]