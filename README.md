# init variable
reset：
随机生成traffic和linkstate(全部为1，利用率目前为0)，是否需要记录初始traffic
写traffic， 根据weights写routing，进入omnet，得到链路利用率

step：
拿到action更新weights
weights更新routing，比较是否有跟linkstate不符的链路：
    符合：omnet，得到loss，计算utility，更新linkstate，done=0，reward=sum（1-cost）
    不符合：生成新的traffic，linkstate=1，done=1，reward=-1

符合：
    生成新的，断掉一条链路，生成新的graph
    生成新的traffic，routing



omnet部分：
输出链路的loss信息


如何生成链路断掉的情况:
1 linkstate = matrix , 选取一边变为0，视作链路没有了
2 视作graph 中实际有条链路断掉了
