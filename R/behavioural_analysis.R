rm(list=ls())
inv.logit <- function(x) exp(x) / (1 + exp(x))
library(lme4)


# LOAD DATA ---------------------------------------------------------------

behavioural.path <- paste('/home/lau/projects/functional_cerebellum/',
                            'scratch/behavioural_data', sep='')
subjects.df <- data.frame(
    subject=c('0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008',
              '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016',
              '0017', '0018', '0019', '0020', '0021', '0022', '0023', '0024',
              '0026', '0027', '0028', '0029', '0030', '0031'),
    date=c('20210810_000000', '20210804_000000', '20210802_000000',
           '20210728_000000', '20210728_000000', '20210728_000000',
           '20210728_000000', '20210730_000000', '20210730_000000',
           '20210730_000000', '20210730_000000', '20210802_000000',
           '20210802_000000', '20210802_000000', '20210804_000000',
           '20210804_000000', '20210805_000000', '20210805_000000',
           '20210805_000000', '20210806_000000', '20210806_000000',
           '20210806_000000', '20210809_000000', '20210809_000000',
           '20210810_000000', '20210810_000000', '20210817_000000',
           '20210817_000000', '20210817_000000', '20210825_000000'),
    time.stamp=c('075323', '075218', '133323',  '074414', '110852', '134437',
                 '153731', '084401', '111206',  '131907', '163628', '085256',
                 '100241', '160424', '124445',  '153734', '075941', '103918',
                 '122553', '095715', '135334',  '155547', '081423', '103941',
                 '101724', '122355', '102259',  '131300', '161750', '085032')
)

n.subjects <- dim(subjects.df)[1]

for(subject.index in 1:n.subjects)
{
    row <- subjects.df[subject.index, ]
    path <- paste(behavioural.path, row$subject, row$date, sep='/')
    files <- dir(path)
    for(file in files)
    {
        pattern <- paste(row$time.stamp, 'data', sep='_')
        if(grepl(pattern, file)) data.file <- file
    }
    full.path <- paste(path, data.file, sep='/')
    
    if(subject.index == 1)
        { 
            data <- read.csv(full.path)
            data$subject <- subjects.df$subject[subject.index]
        } else
        {
            this.data <- read.csv(full.path)
            this.data$subject <- subjects.df$subject[subject.index]
            data <- rbind(data, this.data)
        }
    
}

data <- subset(data, select=-X)

# REDUCE DATA TO TARGET TRIALS --------------------------------------------

data.target <- subset(data, data$response != 'None')
data.target$trigger <- factor(data.target$trigger)
data.target$current <- as.numeric(as.character(data.target$current))
data.target$response <- factor(data.target$response)
data.target$trial_number <- factor(data.target$trial_number)
data.target$sequence_number <- factor(data.target$sequence_number)
data.target$block_number <- factor(data.target$block_number)
data.target$response_time <- as.numeric(as.character(data.target$response_time))
data.target$jitter <- NA
data.target$stimulation_type <- NA
data.target$response_type <- NA

# ESTIMATE MEAN TIME FOR STEADY TRIALS ------------------------------------

# data$trial_time <- data$trial_time * 1000 # base on milliseconds?

triggers <- c(19, 21, 25)
n.triggers <- length(triggers)

means <- numeric(n.triggers)
for(trigger.index in 1:n.triggers)
{
    trigger <- triggers[trigger.index]
    means[trigger.index] <-
        mean(data$trial_time[data$trigger == trigger])
}

jitter.triggers <- c(19, 35)
n.trials <- dim(data)[1]
n.trials.per.sequence <- 7
variance <- numeric(n.trials / n.trials.per.sequence)

variance.index <- 1
for(trial.index in 1:n.trials)
{
    if(data$trigger[trial.index] == jitter.triggers[1] |
       data$trigger[trial.index] == jitter.triggers[2])
    {
        these.indices <- trial.index:(trial.index + 2)
        these.times <- data$trial_time[these.indices]
        variance[variance.index] <- sum((these.times - means)^2)
        variance.index <- variance.index + 1
    }
}

data.target$variance <- variance

# CODE CONDITIONS BASED ON TRIGGERS ---------------------------------------

data.target$jitter[data.target$trigger == '81'] <- '0%'
data.target$jitter[data.target$trigger == '97'] <- '15%' 
data.target$jitter[data.target$trigger == '144'] <- '0%'
data.target$jitter[data.target$trigger == '160'] <- '15%'

data.target$stimulation_type[data.target$trigger == '81'] <- 'weak'
data.target$stimulation_type[data.target$trigger == '97'] <- 'weak' 
data.target$stimulation_type[data.target$trigger == '144'] <- 'omission'
data.target$stimulation_type[data.target$trigger == '160'] <- 'omission'

data.target$response_type[data.target$stimulation_type == 'weak' &
                          data.target$response == 'yes'] <- 'hit'
data.target$response_type[data.target$stimulation_type == 'weak' & 
                          data.target$response == 'no'] <- 'miss'
data.target$response_type[data.target$stimulation_type == 'omission' & 
                          data.target$response == 'yes'] <- 'false.alarm'
data.target$response_type[data.target$stimulation_type == 'omission' & 
                          data.target$response == 'no'] <- 'correct.rejection'

data.target$correct[data.target$response_type == 'hit' | 
                    data.target$response_type == 'correct.rejection'] <- 1
data.target$correct[data.target$response_type == 'miss' | 
                        data.target$response_type == 'false.alarm'] <- 0

data.target$jitter <- factor(data.target$jitter)
data.target$stimulation_type <- factor(data.target$stimulation_type)
data.target$response_type <- factor(data.target$response_type)


# D PRIME DATA FRAME ------------------------------------------------------

n.subjects <- length(levels(data.target$subject))

d.prime <- numeric()
currents <- numeric()

for(subject.index in 1:n.subjects)
{
    subject <- subjects.df$subject[subject.index]
    data.subject <- data.target[data.target$subject == subject, ]
    for(jitter.level in levels(data.subject$jitter))
    {
        this.data <- data.subject[data.subject$jitter == jitter.level, ]
        hit.rate <- 
            sum(this.data$correct[this.data$stimulation_type == 'weak']) /
            sum(this.data$stimulation_type == 'weak')
        false.alarm.rate <- 
            1 -
            sum(this.data$correct[this.data$stimulation_type == 'omission']) /
            sum(this.data$stimulation_type == 'omission')
        if(false.alarm.rate == 0) false.alarm.rate <- 1 / 
            sum(this.data$stimulation_type == 'omission') 
        this.d.prime <- qnorm(hit.rate) - qnorm(false.alarm.rate)
        d.prime <- c(d.prime, this.d.prime)
        currents <- c(currents, min(this.data$current, na.rm=TRUE))

    }
}

data.d.prime <- data.frame('d.prime'=d.prime, 'current'=currents)
data.d.prime$subject <- rep(levels(data.target$subject), each=2)
data.d.prime$jitter <- levels(data.target$jitter)


# FIT SINGLE SUBJECT MODELS -----------------------------------------------
# 
# models <- list()
# performance <- numeric(n.subjects)
# 
# for(subject.index in 1:n.subjects)
# {
#     subject <- subjects.df$subject[subject.index]
#     this.data <- data.target[data.target$subject == subject, ]
#     performance[subject.index] <- sum(this.data$correct) / 
#         length(this.data$correct)
#     model <- glm(correct ~ stimulation_type * jitter, data=this.data,
#                  family='binomial')
#     models[[length(models) + 1]] <- summary(model)
# }


# FIT BIG GLMER -----------------------------------------------------------

# mm <- glmer(correct ~ stimulation_type * jitter + (1 | subject),
#             data=data.target, family='binomial', verbose=TRUE)
# 
# mm.rt <- lmer(log(response_time) ~ stimulation_type * jitter + (1 | subject),
#               data=data.target, verbose=TRUE)

# remove ceiling and chance performers??


# VARIANCE TEST -----------------------------------------------------------


## convergence issues
# control <- glmerControl(optCtrl=list(Xtol_Rel=1e-8, FtolAbs=1e-8))
# optCtrl <- list(FtolAbs=1e-10, FtolRel=1e-20, XtolRel=1e-11)
# control <- glmerControl(optCtrl=optCtrl)
control <- glmerControl()
control$checkConv$check.conv.grad$tol <- 0.004 ## this "works"...
# data.target$variance.c <- scale(data.target$variance.c, scale=FALSE)
full.model <- glmer(correct ~ stimulation_type * variance +
                   (stimulation_type * variance | subject),
                    data=data.target, family='binomial', verbose=2,
                   control=control)

model.no.int <- glmer(correct ~ stimulation_type + variance + 
                          (stimulation_type * variance | subject),
                      data=data.target, family='binomial', verbose=2,
                      control=control) ## winning model

model.no.stim <- glmer(correct ~ variance + 
                           (stimulation_type * variance | subject),
                       data=data.target, family='binomial', verbose=2,
                       control=control)

model.no.var <- glmer(correct ~ stimulation_type + 
                           (stimulation_type * variance | subject),
                       data=data.target, family='binomial', verbose=2,
                      control=control)

model.null <- glmer(correct ~ 1 + 
                        (stimulation_type * variance | subject),
                    data=data.target, family='binomial', verbose=2,
                    control=control)


# WINNING MODEL -----------------------------------------------------------

group.coefs <- fixef(model.no.int)
omission.accuracy <- inv.logit(group.coefs[1])
weak.accuracy <- inv.logit(sum(group.coefs[1:2]))
max.variance <- max(data.target$variance)
non.zero.variances <- 
    data.target$variance[data.target$variance > median(data.target$variance)]
mean.variance <- mean(non.zero.variances)

omission.accuracy.max.variance <- 
    inv.logit(group.coefs[1] + max.variance * group.coefs[3])
weak.accuracy.max.variance <- 
    inv.logit(sum(group.coefs[1:2]) + max.variance * group.coefs[3])
omission.accuracy.mean.variance <- 
    inv.logit(group.coefs[1] + mean.variance * group.coefs[3])
weak.accuracy.mean.variance <- 
    inv.logit(sum(group.coefs[1:2]) + mean.variance * group.coefs[3])


print(omission.accuracy)
print(weak.accuracy)
print(omission.accuracy.max.variance)
print(weak.accuracy.max.variance)
print(omission.accuracy.mean.variance)
print(weak.accuracy.mean.variance)

# PLOT VARIANCE TEST  (limited range) -----------------------------------------

jpeg('/home/lau/projects/functional_cerebellum/scratch/figures/behaviour.jpeg')

par(lwd=3, font.lab=2, font.axis=2)
xlim <- range(data.target$variance)


intercept <- fixef(full.model)[1]
stim.type <- fixef(full.model)[2]
slope <- fixef(full.model)[3]
slope.change <- fixef(full.model)[4]

x <- seq(0, max(data.target$variance), length.out=100)
y1 <- inv.logit(intercept + slope*x)
y2 <- inv.logit(intercept + stim.type + (slope+slope.change)*x)

plot(x, y1, type='l', xlab='Variance of last three stimulations (s²)',
     ylab='Propotion correct', ylim=c(0.70, 1.00),
     main='Behavioural performance')
lines(x, y2, col='red')
legend('topright', legend=c('Omission', 'Weak'), lty=1, col=c('black', 'red'),
       text.font=2)

dev.off()

x <- seq(0, 3, length.out=1000)
y1 <- inv.logit(intercept + slope*x)
y2 <- inv.logit(intercept + stim.type + (slope+slope.change)*x)
plot(x, y1, type='l', xlab='Variance of last three stimulations (s²)',
     ylab='Propotion correct', ylim=c(0.00, 1.00))
lines(x, y2, col='red')
legend('topright', legend=c('Omission', 'Weak'), lty=1, col=c('black', 'red'),
       text.font=2)


# PLOT SINGLE SUBJECTS AS WELL --------------------------------------------


par(lwd=3, font.lab=2, font.axis=2, mfrow=c(2, 1))
xlim <- range(data.target$variance)


intercept <- fixef(full.model)[1]
stim.type <- fixef(full.model)[2]
slope <- fixef(full.model)[3]
slope.change <- fixef(full.model)[4]

x <- seq(0, max(data.target$variance), length.out=100)
# x <- seq(0, 2, length.out=100)
# xlim <- c(0, 2)
y1 <- inv.logit(intercept + slope*x)
y2 <- inv.logit(intercept + stim.type + (slope+slope.change)*x)

subject.effects <- ranef(full.model)$subject
n.subjects <- dim(subject.effects)[1]

plot(x, y1, type='n', xlab='Variance of last three stimulations (s²)',
     ylab='Propotion correct', ylim=c(0.4, 1.00), lwd=10, main='Omission')

for(subject.index in 1:n.subjects)
{
    subject.intercept <- subject.effects[subject.index, 1]
    subject.slope <- subject.effects[subject.index, 3]
    y <- inv.logit(intercept + subject.intercept + (slope + subject.slope) * x)
    lines(x, y, col='red')
    if((slope + subject.slope) > 0) print(subjects.df$subject[subject.index])
}

lines(x, y1, lwd=10)
## weak
plot(x, y2, type='n', xlab='Variance of last three stimulations (s²)',
     ylab='Propotion correct', ylim=c(0.4, 1.00), lwd=10, main='Weak')

for(subject.index in 1:n.subjects)
{
    subject.intercept <- subject.effects[subject.index, 1]
    subject.stim.type <- subject.effects[subject.index, 2]
    subject.slope <- subject.effects[subject.index, 3]
    subject.slope.change <- subject.effects[subject.index, 4]
    
    y <- inv.logit(intercept + subject.intercept + stim.type + 
                       subject.stim.type +
                       (slope + subject.slope + 
                            slope.change + subject.slope.change) * x)
    lines(x, y, col='red')
}

lines(x, y2, lwd=10)


# GET DATA FRAME OF SUBJECT SLOPES ----------------------------------------

mean.slope <- fixef(model.no.int)[3]

subject.slopes <- data.frame(weak=numeric(n.subjects),
                             omission=numeric(n.subjects),
                             subject=subjects.df$subject)

subject.estimates <- ranef(model.no.int)[1]$subject

for(subject.index in 1:n.subjects)
{
    subject.slopes$omission[subject.index] <- 
        mean.slope + subject.estimates$variance[subject.index]
    subject.slopes$weak[subject.index] <- 
        mean.slope + subject.estimates$variance[subject.index] +
        subject.estimates$`stimulation_typeweak:variance`[subject.index]
}


# WRITE SUBJECT SLOPES TO CSV ---------------------------------------------

path <- '/home/lau/projects/functional_cerebellum/scratch/behavioural_data'
filename <- 'subject_slopes.csv'

write.csv(subject.slopes, file=paste(path, filename, sep='/'), row.names=FALSE)

# EVOKED-BEHAVIOUR CORRELATIONS -------------------------------------------





# PLAY AROUND WITH PEAK CEREBELLAR VALUES ---------------------------------
# do one with evoked responses
# copied from python
# 
# good.subject.indices <- c(1:5, 7:9, 12:27, 29:30)
# # good.subject.indices <- c(1:5, 7:9, 12:23, 25, 27, 29:30)
# 
# nan <- NaN
# 
# w0 <- c(2.33014359e-14, 9.56376769e-15, 4.47699942e-15, 1.63622985e-15,
#            3.62656775e-15,            nan, 4.63585452e-15, 3.07619530e-14,
#            2.80905073e-15,            nan,            nan, 1.13409114e-14,
#            9.57100474e-15, 6.75477816e-15, 2.12748977e-14, 4.05020933e-14,
#            1.05259948e-15, 2.56007577e-14, 4.18590172e-15, 7.02164055e-15,
#            2.08952271e-14, 6.80311368e-15, 1.62215635e-14, 1.15949620e-14,
#            5.35746010e-15, 2.68529731e-15, 3.24081243e-15,            nan,
#            8.39723392e-15, 5.79607493e-15)
#     
# w15 <- c(7.11959822e-15, 2.15327261e-14, 3.56867432e-15, 1.24704832e-14,
#            7.32591681e-15,            nan, 6.92249111e-15, 1.78201570e-14,
#            1.81008781e-15,            nan,            nan, 1.50704983e-14,
#            2.25879021e-14, 4.75702936e-15, 3.12593714e-14, 5.22912470e-14,
#            6.54641801e-16, 5.86703063e-14, 3.50437995e-14, 1.37176435e-14,
#            1.63483489e-15, 2.34349537e-15, 6.19372556e-15, 9.61844875e-15,
#            1.15738802e-14, 2.72936225e-14, 6.39106264e-15,            nan,
#            1.32437709e-14, 1.03147014e-14)
# 
# cerebellar.weak <- w0 - w15
# 
# cerebellar.omission <- c(-0.0692599 , -0.02860739, -0.0070418 ,  0.04176447, -0.02920996,
#                          nan, -0.0032061 ,  0.02474657, -0.04801918,         nan,
#                          nan,  0.00883006, -0.01156769, -0.00969528, -0.05480513,
#                          -0.02789011, -0.01387964, -0.02019667,  0.01245323, -0.00054537,
#                          -0.01416469,  0.01343061,  0.00235587,  0.00701085,  0.02761748,
#                          -0.0097693 ,  0.05614039,         nan, -0.05033996, -0.01657967)
# 
# # plot(subject.slopes$omission[good.subject.indices], cerebellar.omission)
# #
# omission <- data.frame(cerebellar=cerebellar.omission[good.subject.indices],
#                        slope=subject.slopes$omission[good.subject.indices])
# 
# 
# weak <- data.frame(cerebellar=cerebellar.weak[good.subject.indices],
#                    slope=subject.slopes$weak[good.subject.indices],
#                    d.prime=data.d.prime$d.prime[data.d.prime$jitter == '0%'][good.subject.indices])
# 
# hist(weak$slope)
# hist(weak$cerebellar)
# hist(weak$d.prime)
# plot(cerebellar ~ slope, data=omission)
# abline(lm(cerebellar ~ slope, data=omission))
# 
# plot(cerebellar ~ slope, data=weak)
# abline(lm(cerebellar ~ slope, data=weak))
# 
# # plot(cerebellar ~ d.prime, data=weak)
# # abline(lm(cerebellar ~ d.prime, data=weak))
# 
# print(cor.test(weak$cerebellar, weak$slope, method='spearman'))
# print(cor.test(weak$cerebellar, weak$d.prime, method='pearson'))
# RUN FULL MODEL RESPONSE TIMES -------------------------------------------
# 
# control <- lmerControl()
# 
# 
# full.model <- lmer(log(response_time) ~ stimulation_type * variance + 
#                         (variance | subject),
#                     data=data.target, verbose=2,
#                     control=control, REML=FALSE)
# 
# model.no.int <- lmer(log(response_time) ~ stimulation_type + variance + 
#                           (stimulation_type * variance | subject),
#                       data=data.target, verbose=2,
#                       control=control, REML=FALSE)
# 
# model.no.stim <- lmer(log(response_time) ~ variance + 
#                            (stimulation_type * variance | subject),
#                        data=data.target, verbose=2,
#                        control=control, REML=FALSE)
# 
# model.no.var <- glmer(log(response_time) ~ stimulation_type + 
#                           (stimulation_type * variance | subject),
#                       data=data.target, verbose=2,
#                       control=control, REML=FALSE)
# 
# model.null <- glmer(log(response_time) ~ 1 + 
#                         (stimulation_type * variance | subject),
#                     data=data.target, verbose=2,
#                     control=control, REML=FALSE)
