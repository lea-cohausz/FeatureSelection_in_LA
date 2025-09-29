library(bnlearn)

# Load the data
data <- read.csv("data.csv")

# Convert all variables to factors (discrete)
data[] <- lapply(data, as.factor)

# Specify the blacklist: prohibit arcs ending at "gender"
vars <- setdiff(names(data), "gender")
blacklist <- data.frame(from = vars, to = rep("gender", length(vars)))

# Learn the network structure with background knowledge
# Use BDe or BIC score for discrete data ("bde" or "bic")
obj_hc <- hc(data, score = "aic", blacklist = blacklist)

desc_target = descendants(obj_hc, "target")

cat(desc_target, sep=" ")