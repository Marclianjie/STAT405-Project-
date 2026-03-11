library(tidyverse)

getwd()
setwd("W:/UBC/STAT_405/STAT405-Project-")

data_affect <- read_csv("Data/Primary/main_data.csv")




data_alt <- read_tsv("Data/0045_kullar_ts.tsv", show_col_types = FALSE)

# basic checks
head(data_alt)
dim(data_alt)
names(data_alt)
str(data_alt)
summary(data_alt)
colSums(is.na(data_alt))

# number of participants / rows per participant
data_alt %>%
  summarise(
    n_obs = n(),
    n_subjects = n_distinct(id)
  )