"""

likelihook解释 参见: https://alexanderetz.com/2015/04/15/understanding-bayes-a-look-at-the-likelihood/
最后的图可以解释为:
以 head=7为例 

当投掷 10 次硬币, 获取 7 次正面的二项式模型里
(p=0.69)->Binomial(10,p)|>d->pdf(d,data=7)
返回的 pdf 为最大值

当然 p=0.50 的硬币也有扔出 7 次正面的机会, 但是概率比 p=0.69小. 

如果拿 d1=Binomail(10,0.69) 和 d2=Binomail(10,0.50) 两个分布做为模型
d1  获取 7 次正面的机会比较大.  

在这里有很主观的干扰. 就是对硬币的 p=0.5 的主观先验概率的认识很强烈,以至于我们都认为这个概率不会变,实际上没有任何数据
支持所有的硬币的二项式分布的概率为p=0.5.  
当我们扔 100 次硬币, 47 次这正面, 53 次反面的 极大可能是由 p=0.5的硬币导致的结果. 这是一种猜测, 最大可能的统计模型就是极大似然率
"""


using Distributions,GLMakie

tri=10  #试验次数

"""
input_data(data)
个高阶函数, 第一个参数为数据, 在硬币实验中为正面的次数
第二个参数为抛投硬币正面的概率

返回值为在某个固定结果下, 各种不同正面概率参数获取的概率密度值. 

!!! notice
    这就是最大似然估计的含义, 给定一组数据, 我们想要知道的是某个模型的哪个参数可以最大可能的产生数据
    对于硬币试验,我们给定一组正面的数据, 想要知道的是二项式分布的哪一个参数能最大可能的产生这个结果
## 



"""
function input_data(data)
   return  (p)->Binomial(10,p)|>d->pdf(d,data)
end


probrange=range(0.0,1.0,50)

params=0:10

fig=Figure()


for  i in params
     local row= div(i,3)
     local col=rem(i,3)
    local ax=Axis(fig[row+1,col])
    local likelihood=input_data(i)
    
    max_likelihood=likelihood.(probrange)|>maximum
    x=undef
    for i in probrange
        if likelihood(i)==max_likelihood
            x=round(i,digits=2)
            #@show x
        end
    end
    lines!(ax,probrange,likelihood.(probrange),label="head=$(i)")
    vlines!(ax,[x],label="p=$(x)")
    axislegend()
end

fig
#save("likelihood-function-for-n-heads-in-10-tosses.png",fig)

