########################################
## AMAZON EMPLOYEE ACCESS PREDICTIONS ##
########################################

# libraries
library(tidymodels)
library(tidyverse)
library(vroom)


# read in data
train_data <- vroom("/Users/eliseclark/Documents/Fall 2025/Stat 348/AmazonEmployeeComp/amazon-employee-access-challenge/train.csv")


###### EDA ######
ggplot(data = train_data, aes(x = ACTION)) + 
  geom_bar(fill = "plum3") + 
  xlab("Action") + ylab("Frequency") + 
  ggtitle("Amazon Employee Access Counts")


table(train_data$ACTION, train_data$ROLE_FAMILY)


# 1050 is correct!


















