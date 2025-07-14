library(spectral.methods) # runs with R 3.6.3
library(tidyverse)
library(janitor)

data <- read_csv("C:/Users/bahrens/Desktop/EasyHybrid/projects/RsoilComponents/data/RESP_07_08_09_10_prep.csv")

# Replace all values smaller -100 numeric values with NA
data <- data %>% mutate(across(where(is.numeric), ~ if_else(. < -100, NA_real_, .)))

# Find the last good row (not all NAs)
last_good_row <- which(!apply(is.na(data), 1, all)) %>% last()

data <- data[1:last_good_row, ]

# Drop columns that are all NA
data <- data %>% select(where(~ !all(is.na(.))))

# Clean column names
data <- data %>% clean_names()

colnames(data)

cols_to_fill <- c("cham_temp", "temp_surface", "temp_2_cm", "temp_10_cm", "temp_20_cm", "moisture")

nrow(data)

summary(data)

m2fill <- as.matrix(data[ , cols_to_fill])

obj <- m2fill
obj[] <- NA
colnames(obj) <- paste0(cols_to_fill, "_filled")
for(i in cols_to_fill) {
  print(i)
  obj[,paste0(i, "_filled")] <- gapfillSSA(series = m2fill[,i], M = 120
                                           ,remove.infinite = TRUE, seed=1983
                                           , fill.margins = FALSE
                                           , open.plot = TRUE, plot.results = TRUE, plot.progress = TRUE)$filled.series
}

filled_data <- bind_cols(data, obj)

write_csv(filled_data, "C:/Users/bahrens/Desktop/EasyHybrid/projects/RsoilComponents/data/RESP_07_08_09_10_filled.csv")
