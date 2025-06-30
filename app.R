# app.R
library(shiny)
library(tidyverse)
library(zoo)
library(broom)
library(lubridate)

# Load and prepare data (pre-load once to keep app responsive)
load_data <- function() {
  spc <- read_csv("^GSPC_mon.csv") %>%
    rename(Date = 1) %>%
    mutate(Date = as.Date(Date)) %>%
    select(Date, SPC = Close)
  
  spe <- read_csv("RSP_mon.csv") %>%
    rename(Date = 1) %>%
    mutate(Date = as.Date(Date)) %>%
    select(Date, SPE = Close)
  
  price_df <- full_join(spc, spe, by = "Date") %>%
    arrange(Date) %>%
    drop_na()
  
  returns_df <- price_df %>%
    mutate(
      SPC_ret = (SPC / lag(SPC)) - 1,
      SPE_ret = (SPE / lag(SPE)) - 1,
      Innovation = SPC_ret - SPE_ret
    ) %>% drop_na()
  
  ff <- read_csv("F-F_Research_Data_5_Factors_2x3.csv", skip = 3) %>%
    rename(Date = 1) %>%
    filter(str_detect(Date, "^\\d{6}")) %>%
    mutate(Date = ymd(paste0(Date, "01")),
           across(-Date, ~ suppressWarnings(as.numeric(.x) / 100)))
  
  ff_mom <- read_csv("F-F_Momentum_Factor.csv", skip = 13, n_max = 1180) %>%
    rename(Date = 1, Mom = 2) %>%
    mutate(Date = ymd(paste0(Date, "01")), Mom = suppressWarnings(as.numeric(Mom) / 100))
  
  ff_all <- left_join(ff, ff_mom, by = "Date")
  
  merged_df <- left_join(returns_df, ff_all, by = "Date") %>% drop_na()
  
  excess_df <- merged_df %>%
    transmute(
      Date,
      SPC = SPC_ret - RF,
      SPE = SPE_ret - RF,
      Innovation = Innovation - RF,
      `Mkt-RF`, SMB, HML, RMW, CMA, Mom
    )
  
  return(excess_df)
}

compute_rolling_betas_and_alpha <- function(df, portfolio_col, window = 36,
                                            beta_MKT = FALSE,
                                            beta_SMB = TRUE,
                                            beta_HML = TRUE,
                                            beta_RMW = TRUE,
                                            beta_CMA = TRUE,
                                            beta_Momentum = FALSE) {
  req(nrow(df) >= window)
  
  factor_flags <- list(
    `Mkt-RF` = beta_MKT,
    SMB = beta_SMB,
    HML = beta_HML,
    RMW = beta_RMW,
    CMA = beta_CMA,
    Mom = beta_Momentum
  )
  
  selected_factors <- names(Filter(identity, factor_flags))
  if (length(selected_factors) == 0) return(tibble())
  
  result_list <- list()
  
  for (i in seq(window, nrow(df))) {
    window_df <- df[(i - window + 1):i, ]
    
    y <- window_df[[portfolio_col]]
    X <- window_df[, selected_factors, drop = FALSE]
    X <- cbind(Intercept = 1, X)
    
    model <- tryCatch({
      lm(y ~ . -1, data = as.data.frame(X))
    }, error = function(e) return(NULL))
    
    if (is.null(model)) next
    
    model_summary <- summary(model)
    coefs <- coef(model)
    tstats <- coef(model_summary)[, "t value"]
    pvals <- coef(model_summary)[, "Pr(>|t|)"]
    
    alpha <- coefs["Intercept"]
    alpha_tstat <- tstats["Intercept"]
    alpha_pval <- pvals["Intercept"]
    r_squared <- model_summary$r.squared
    
    downside_std <- sd(y[y < 0])
    sortino <- if (!is.na(downside_std) && downside_std > 0) mean(y) / downside_std else NA_real_
    
    row <- list(
      Date = window_df$Date[window],
      alpha = alpha,
      alpha_tstat = alpha_tstat,
      alpha_pval = alpha_pval,
      r_squared = r_squared,
      sortino = sortino
    )
    
    for (factor in selected_factors) {
      beta_name <- paste0("beta_", factor)
      row[[beta_name]] <- coefs[factor]
    }
    
    result_list[[length(result_list) + 1]] <- row
  }
  
  result_df <- bind_rows(result_list)
  return(as_tibble(result_df))
}

excess_df <- load_data()

# UI
ui <- fluidPage(
  titlePanel("Rolling Factor Model Analysis"),
  sidebarLayout(
    sidebarPanel(
      selectInput("portfolio", "Portfolio:", choices = c("SPC", "SPE", "Innovation"), selected = "Innovation"),
      sliderInput("window", "Rolling Window (months):", min = 6, max = 60, value = 36, step = 6),
      checkboxGroupInput("factors", "Select Factors:",
                         choices = c("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"),
                         selected = c("SMB", "HML", "RMW", "CMA", "Mom"))
    ),
    mainPanel(
      plotOutput("betaPlot"),
      tableOutput("summaryTable")
    )
  )
)

# Server
server <- function(input, output) {
  results <- reactive({
    compute_rolling_betas_and_alpha(
      df = excess_df,
      portfolio_col = input$portfolio,
      window = input$window,
      beta_MKT = "Mkt-RF" %in% input$factors,
      beta_SMB = "SMB" %in% input$factors,
      beta_HML = "HML" %in% input$factors,
      beta_RMW = "RMW" %in% input$factors,
      beta_CMA = "CMA" %in% input$factors,
      beta_Momentum = "Mom" %in% input$factors
    )
  })
  
  output$betaPlot <- renderPlot({
    validate(
      need(nrow(results()) > 0, "No results to plot. Check selected window and factors.")
    )
    plot_df <- results() %>%
      pivot_longer(-Date, names_to = "Metric", values_to = "Value") %>%
      filter(Metric %in% c("alpha", paste0("beta_", input$factors)))
    
    ggplot(plot_df, aes(x = Date, y = Value, color = Metric)) +
      geom_line() +
      labs(title = paste0(input$window, "-Month Rolling Alpha & Betas"), y = "Estimate") +
      theme_minimal()
  })
  
  output$summaryTable <- renderTable({
    validate(
      need(nrow(results()) > 0, "No results to summarize. Try different inputs.")
    )
    selected_df <- results()
    summary_df <- selected_df %>%
      select(-Date) %>%
      summarise_all(list(mean = mean, sd = sd), na.rm = TRUE) %>%
      pivot_longer(everything(), names_to = c("Metric", "Stat"), names_sep = "_") %>%
      pivot_wider(names_from = Stat, values_from = value) %>%
      mutate(across(where(is.numeric), ~ round(.x, 4)))
    
    return(summary_df)
  })
}

shinyApp(ui = ui, server = server)