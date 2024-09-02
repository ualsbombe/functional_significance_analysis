rm(list=ls())
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
        pattern <- paste(row$time.stamp, 'staircase', sep='_')
        if(grepl(pattern, file)) data.file <- file
    }
    full.path <- paste(path, data.file, sep='/')
    
    if(subject.index == 1)
    { 
        data <- read.csv(full.path, header=FALSE)
        n.rows = dim(data)[1]
        data <- data[2:n.rows,]
        data$subject <- subjects.df$subject[subject.index]
    } else
    {
        this.data <- read.csv(full.path, header=FALSE)
        n.rows = dim(this.data)[1]
        this.data <- this.data[2:n.rows,]
        this.data$subject <- subjects.df$subject[subject.index]
        
        data <- rbind(data, this.data)
    }
    
}

data <- subset(data, select=-V5)
data <- subset(data, select=-V4)
colnames(data) <- c('current', 'response', 'trial.no', 'subject')
data$current <- as.numeric(data$current)
data$response <- factor(data$response)
data$trial.no <- as.numeric(data$trial.no)
data$trial.no <- data$trial.no + 1
data$subject <- factor(data$subject)
data$correct <- NA



# GET CORRECT AND WRONG TRIALS --------------------------------------------

for(row.index in 1:dim(data)[1])
{
    if(row.index %% 40 > 0)
    {
        if(data$current[row.index + 1] < data$current[row.index])
        {
            data$correct[row.index] <- 1    
        } else data$correct[row.index] <- 0
        
    }
}

# FIT MODEL ---------------------------------------------------------------

model <- glmer(correct ~ current + (current | subject), 
               data=data, family='binomial',
               verbose=2)


# WRITE IT -----------------------------------------------------------------
path <- '/home/lau/projects/functional_cerebellum/scratch/behavioural_data'
filename <- 'staircase_group_coefficients.csv'
write.csv(t(fixef(model)), file=paste(path, filename, sep='/'),
          row.names=FALSE)
filename <- 'staircase_subject_coefficients.csv'
write.csv(ranef(model)$subject, file=paste(path, filename, sep='/'),
          row.names=FALSE)



