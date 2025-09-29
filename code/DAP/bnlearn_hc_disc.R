library(bnlearn)

# Load the data
data <- read.csv("data.csv")

# Convert all variables to factors (discrete)
data[] <- lapply(data, as.factor)

# Specify the blacklist: prohibit arcs ending at "gender"
# Specify blacklist: all arcs ending at "gender"
vars <- setdiff(names(data), "gender")
blacklist <- data.frame(from = vars, to = rep("gender", length(vars)))
root_vars <- c('gender', 'EDU_FATHER', 'EDU_MOTHER',
       'STRATUM', 'SISBEN', 'INTERNET', 'TV', 'COMPUTER',
       'WASHING_MCH', 'MIC_OVEN', 'CAR', 'DVD', 'FRESH', 'PHONE', 'MOBILE','REVENUE')
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
# Find the Markov Blanket of the target variable
mb_target <- mb(obj_hc, "target")

# Output the Markov Blanket
cat(mb_target, sep = " ")