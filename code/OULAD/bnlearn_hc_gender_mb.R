library(bnlearn)

# Load the data
data <- read.csv("data.csv")

# Convert all variables to factors (discrete)
data[] <- lapply(data, as.factor)

# Specify blacklist: all arcs ending at "gender"
vars <- setdiff(names(data), "gender")
blacklist <- data.frame(from = vars, to = rep("gender", length(vars)))
root_vars <- c("age_band", "disability", "imd_band", "region_East_Anglian_Region", "region_East_Midlands_Region", "region_Ireland", "region_North_Region", "region_North_Western_Region", "region_Scotland", "region_South_East_Region", "region_South_Region", "region_South_West_Region", "region_Wales", "region_West_Midlands_Region", "region_Yorkshire_Region")
leaf_var <- "target"
for (v in root_vars) {
  blacklist <- rbind(
    blacklist,
    data.frame(from = setdiff(names(data), v), to = v)
  )
}

# Add leaf node constraint: no arcs FROM leaf_var
blacklist <- rbind(
  blacklist,
  data.frame(from = leaf_var, to = setdiff(names(data), leaf_var))
)

blacklist <- unique(blacklist)

# Learn the network structure with background knowledge
# Use BDe or BIC score for discrete data ("bde" or "bic")
obj_hc <- hc(data, score = "aic", blacklist = blacklist)

# Find the Markov Blanket of the target variable
mb_target <- mb(obj_hc, "gender")

# Output the Markov Blanket
cat(mb_target, sep = " ")