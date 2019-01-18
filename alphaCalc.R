alphaScores <- function(matrix,f,S){
  for (i in 2:5){
    fairnessScore<- matrix[c(f),i]
    accuracy<- matrix[c(1),i]
    fairnessScore<- 1-abs(fairnessScore)
    alpha<- (0:100)/100
    combined_score <- (1-alpha)*(accuracy)+(alpha)*(fairnessScore)
    if (i==2){
      plot(alpha, combined_score, col="black", "l",ylim=c(0.65,0.8))
      legend("topleft",c("Adverserial","Prejudice", "Nondebiased", "Ensemble"),col=c("black", "blue","red","green"), fill = c("black", "blue","red","green"))
      
    }
    if (i==3){
      lines(alpha, combined_score, col="blue",lty=2)
    }
    if (i==4){
      lines(alpha, combined_score, col="red",lty=2)
    }
    if (i==5){
      lines(alpha, combined_score, col="green",lty=2)
    }
  } 
  title(main=S, sub="combined_0score= (1-alpha)*(accuracy)+(alpha)*(fairnessScore)")
}