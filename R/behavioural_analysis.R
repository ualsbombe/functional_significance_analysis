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

# REMOVE HIGH AND LOW PERFORMERS ------------------------------------------
# PLOT VARIANCE TEST ------------------------------------------------------

par(lwd=3, font.lab=2, font.axis=2, mfrow=c(2, 1))
xlim <- range(data.target$variance)


intercept <- fixef(full.model)[1]
stim.type <- fixef(full.model)[2]
slope <- fixef(full.model)[3]
slope.change <- fixef(full.model)[4]

x <- seq(0, max(data.target$variance), length.out=100)
y1 <- inv.logit(intercept + slope*x)
y2 <- inv.logit(intercept + stim.type + (slope+slope.change)*x)

plot(x, y1, type='l', xlab='Variance of last three stimulations (s²)',
     ylab='Propotion correct', ylim=c(0.70, 0.95))
lines(x, y2, col='red')
# legend('topright', legend=c('Omission', 'Weak'), lty=1, col=c('black', 'red'),
#        text.font=2)

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
x <- seq(0, 2, length.out=100)
# xlim <- c(0, 2)
y1 <- inv.logit(intercept + slope*x)
y2 <- inv.logit(intercept + stim.type + (slope+slope.change)*x)

subject.effects <- ranef(full.model)$subject
n.subjects <- dim(subject.effects)[1]

plot(x, y1, type='n', xlab='Variance of last three stimulations (s²)',
     ylab='Propotion correct', ylim=c(0, 1.00), lwd=10, main='Omission')

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
     ylab='Propotion correct', ylim=c(0, 1.00), lwd=10, main='Weak')

for(subject.index in 1:n.subjects)
{
    subject.intercept <- subject.effects[subject.index, 1]
    subject.stim.type <- subject.effects[subject.index, 2]
    subject.slope <- subject.effects[subject.index, 3]
    subject.slope.change <- subject.effects[subject.index, 4]
    
    y <- inv.logit(intercept + subject.intercept + stim.type + subject.stim.type + (slope + subject.slope + slope.change + subject.slope.change) * x)
    lines(x, y, col='red')
}

lines(x, y2, lwd=10)


# GET DATA FRAME OF SUBJECT SLOPES ----------------------------------------

mean.slope <- fixef(model.no.int)[3]

subject.slopes <- data.frame(weak=numeric(n.subjects),
                             omission=numeric(n.subjects))

subject.estimates <- ranef(model.no.int)[1]$subject

for(subject.index in 1:n.subjects)
{
    subject.slopes$omission[subject.index] <- 
        mean.slope + subject.estimates$variance[subject.index]
    subject.slopes$weak[subject.index] <- 
        mean.slope + subject.estimates$variance[subject.index] +
        subject.estimates$`stimulation_typeweak:variance`[subject.index]
}

# subject.slopes <- data.frame(slope=c(subject.slopes$weak,
#                                      subject.slopes$omission))

# subject.slopes$type <- factor(c(rep('weak', n.subjects),
#                                 rep('omission', n.subjects)))
# 
# plot(jitter(as.integer(subject.slopes$type)), subject.slopes$slope,
#      xlim=c(0, 3))


# PLAY AROUND WITH PEAK CEREBELLAR VALUES ---------------------------------
# do one with evoked responses
# copied from python

good.subject.indices <- c(1:5, 7:10, 12:30)

# cerebellar.omission <- c(-0.00115931,  0.04012683,  0.02513166,  0.05174428, 
#                          -0.02076979,
#        -0.02795063,  0.03041785,  0.04362067,  0.05811672,  0.02558789,
#        0.0687215 ,  0.00827079, -0.0293969 ,  0.08385569,  0.06076952,
#        0.05528834, -0.03438129,  0.04318867, -0.0091313 ,  0.04282465,
#        0.00326741,  0.03181742, -0.03236426,  0.04787387, -0.0452726 ,
#        0.05254221, -0.02407919, -0.02002526)

# cerebellar.weak <- c(-0.02708759,  0.02797616, -0.02218166, -0.01332885,  0.04542771,
#        0.00452444,  0.02303803,  0.02117814, -0.0162491 ,  0.01630565,
#        -0.00301833,  0.02842695, -0.00538495,  0.05087875,  0.04611307,
#        0.00433935,  0.00472759,  0.00663111,  0.06758594,  0.00436168,
#        0.00573104, -0.00247677,  0.06405189, -0.01341046,  0.04175999,
#        0.08004729,  0.05035557,  0.05765665)

# cerebellar.weak <- c( 0.01565042,  0.06434371,  0.01820711,  0.02298778, -0.02349238,
#                       0.039361  ,  0.03841141, -0.00757248,  0.02642255, -0.02738319,
#                       0.07098991,  0.09036805,  0.04186428, -0.02460145, -0.01427038,
#                       0.0214681 ,  0.04957826,  0.0339404 ,  0.00350517,  0.04075574,
#                       -0.00681289,  0.03060184,  0.00984197,  0.03359656,  0.01059551,
#                       0.01804449, -0.00017876, -0.03953183)

# plot(subject.slopes$omission[good.subject.indices], cerebellar.omission)
# 
# omission <- data.frame(cerebellar=cerebellar.omission,
#                        slope=subject.slopes$omission[good.subject.indices])
# 

weak <- data.frame(cerebellar=cerebellar.weak,
                   slope=subject.slopes$omission[good.subject.indices])

# plot(cerebellar ~ slope, data=omission)
# abline(lm(cerebellar ~ slope, data=omission))

plot(cerebellar ~ slope, data=weak)
abline(lm(cerebellar ~ slope, data=weak))


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
