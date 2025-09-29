library(bnlearn)

# Load the data
data <- read.csv("data.csv")

# Convert all variables to factors (discrete)
data[] <- lapply(data, as.factor)

# Specify the blacklist: prohibit arcs ending at "gender"
vars <- setdiff(names(data), "ge")
blacklist <- data.frame(from = vars, to = rep("ge", length(vars)))

# Learn the network structure with background knowledge
# Use BDe or BIC score for discrete data ("bde" or "bic")
obj_hc <- hc(data, score = "aic", blacklist = blacklist)

# Find the Markov Blanket of the target variable
mb_target <- mb(obj_hc, "ge")

# Output the Markov Blanket
cat(mb_target, sep = " ")