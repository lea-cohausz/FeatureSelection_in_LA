library(bnlearn)

# Load the data
data <- read.csv("data.csv")

# Convert all variables to factors (discrete)
data[] <- lapply(data, as.factor)

# Specify blacklist: all arcs ending at "gender"

root_vars <- c('gender', 'child_age')
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
obj_hc <- hc(data, score = "aic", blacklist = blacklist)

desc_target = descendants(obj_hc, "target")

cat(desc_target, sep=" ")