##-------------------
## Varimax rotation
##-------------------
## k=2
LambdaMat <- matrix(c(1.5,0.1,
                      1.2,0.2,
                      0.2,0.5),
                    nrow=3,ncol=2,byrow=TRUE)
## k=2, Table 9.4.1
LambdaMat <- matrix(c(0.628,0.372,
                      0.696,0.313,
                      0.899,-0.050,
                      0.779,-0.201,
                      0.728,-0.200),
                    nrow=5,ncol=2,byrow=TRUE)
RotMatFct <- function(tht){
  RotMat <- matrix(c(cos(tht),sin(tht),
                  -sin(tht),cos(tht)),
                nrow=2,ncol=2,byrow=TRUE)
  return(RotMat)
}
## Control
RotMatFct(0.2) %*% t(RotMatFct(0.2))
ntht <- 200
thtV <- seq(0,pi/2,len=ntht)
PsiV <- rep(0,ntht)
for (i in 1:ntht){
  ## Delta=Lambda G
  DeltaMat <- LambdaMat %*% RotMatFct(thtV[i])
  ## Communality vector h
  hVec <- sqrt( rowSums( DeltaMat^2 ) )
  #hVec <- sqrt( rowSums( LambdaMat^2 ) )
  ## Normalize by communality
  DMat <- DeltaMat/hVec
  ## Squared loadings
  D2Mat <- DMat^2
  ## Psi function to minimize (objective function)
  MeanD2Mat <- 
    matrix(rep(colMeans(D2Mat),nrow(D2Mat)),nrow=nrow(D2Mat),byrow=TRUE)
  PsiV[i] <- sum( (D2Mat - MeanD2Mat)^2 ) 
}
plot(thtV,PsiV)
## Estimate of tht
thtEst <- thtV[which.max(PsiV)]
## Rotated loadings
LambdaMat %*% RotMatFct(thtEst)
## Compare to results in Mardia Table 9.4.1 page 266 (Example 9.6.1)
thtBook <- (37.6)*2*pi/360
LambdaMat %*% RotMatFct(thtBook)
abline(v=thtBook)
##-----------------------------------------------
## k=3
##------------------------------------------------
LambdaMat <- matrix(c(0.628,0.372,0.9,
                      0.696,0.313,0.9,
                      0.899,-0.050,0,
                      0.779,-0.201,0,
                      0.728,-0.200,0),
                    nrow=5,ncol=3,byrow=TRUE)
LambdaMat%*%t(LambdaMat)

RotMatPairFct <- function(tht,pr){
  if (pr==1){
  RotMat <- matrix(c(cos(tht),sin(tht),0,
                     -sin(tht),cos(tht),0,
                   0,0,1),
                   nrow=3,ncol=3,byrow=TRUE)
  return(RotMat)
  }
  if (pr==2){
  RotMat <- matrix(c(cos(tht),0,sin(tht),
                     0,1,0,
                     -sin(tht),0,cos(tht)),
                   nrow=3,ncol=3,byrow=TRUE)
  return(RotMat)
  }
  if (pr==3){
  RotMat <- matrix(c(1,0,0,
                     0,cos(tht),sin(tht),
                     0,-sin(tht),cos(tht)),
                   nrow=3,ncol=3,byrow=TRUE)
  return(RotMat)
  }
}

## Control
pr <- 2 ; tht <- 0.3
RotMatPairFct(tht,pr)
RotMatPairFct(tht,pr) %*% t(RotMatPairFct(tht,pr))

ntht <- 200
thtV <- seq(0,pi/2,len=ntht)

## Cycles
for (ccl in 1:15){
  ## Pairs of dimensions (in general should be k(k-1)/2 possibilities)
  for (pr in 1:3){
      PsiV <- rep(0,ntht)
      for (i in 1:ntht){
      ## Delta=Lambda G
      DeltaMat <- LambdaMat %*% RotMatPairFct(thtV[i],pr)
      ## Communality vector h
      hVec <- sqrt( rowSums( DeltaMat^2 ) )
      #hVec <- sqrt( rowSums( LambdaMat^2 ) )
      ## Normalize by communality
      DMat <- DeltaMat/hVec
      ## Squared loadings
      D2Mat <- DMat^2
      ## Psi function to minimize (objective function)
      MeanD2Mat <- 
        matrix(rep(colMeans(D2Mat),nrow(D2Mat)),nrow=nrow(D2Mat),byrow=TRUE)
      PsiV[i] <- sum( (D2Mat - MeanD2Mat)^2 ) 
    }
    #plot(thtV,PsiV)
    ## Estimate of tht
    thtEst <- thtV[which.max(PsiV)]
    cat("cycle:",ccl,"theta:",thtEst,"pair:",pr,"Objective function:",max(PsiV),"\n")
    ## Rotated loadings
    LambdaMat <- LambdaMat %*% RotMatPairFct(thtEst,pr)
  }
}

LambdaMat %*% t(LambdaMat)
