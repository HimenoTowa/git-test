git作用：保存一个项目的多次修改的版本

---

vscode 打开文件夹 新建文件test.txt 里面写v0.1

上方Terminal->新建终端

下方Terminal右键，展示到右端

ls 、pwd 、cd .. 、cd .\\文件夹名  这些终端指令都可用

git version （前提是电脑有安装好git）

git config --global user.name "HimenoTowa"

git config --global user.email "136072451@qq.com"

git init （建立.git隐藏文件）

git add test.txt （将要进行项目版本管理的文件加进去）（暂存）

git commit （提交）

将进入vim编辑提交说明（a或i进入编辑模式，第一次提交，先按esc退出编辑模式，然后:wq

git log



修改文件为v0.2

git add . (添加本文件夹所有文件)

git commit -m "第二次提交"  （写提交说明的简化操作）

git log

clear (清屏)



修改文件为v0.3

git add . (添加本文件夹所有文件)

git commit -m "fix(test): change content"  （规范词，可搜git commit风格自行了解）

git log

clear (清屏)

---

现在很多人也不用终端操作了，vscode自带git界面版的操作

---

修改文件为v0.4 这时左边可以看到有一个文件改动的提示 点它

点一下修改的文件 可以看到对比视图 左侧是上一次commit的内容，右侧是目前的内容

左上方有一个√按钮，点一下，就帮你执行git add 和 git commit了

然后在上面输入一下commit消息，也就是说明，回车就行了。

在终端git log一下可以看到信息

---

高级的功能可以安装一个git history diff插件 可以看到所有历史提交（在文件里右键，然后选GitHD:View File History)

---

菜鸟教程里有git功能介绍

---

如何回退到某个版本？

git log

复制一下某一次的commit id

git reset --hard commitId 你复制的Id（鼠标右键粘贴）（最好别hard，否则没有后悔药）

可以看到文件内容变回对应的版本了

git log一下也发现没有后面的了

除了--hard还有--soft和--mixed（默认）

---

如何在不同版本切换？ 用分支branch

分支就是把当前版本复制一份。

commit（界面或代码）

git branch 0.2 （创建一个0.2分支（0.2为分支名））

commit（界面或代码）

git branch 0.3 （创建一个0.3分支）

git branch -a （可看到所有分支）

git checkout（可不断切换分支）

作用：1.切换版本

2.（更重要）可以在主流上继续写代码，也可以在支流上同时写，然后在某一天把这两条分支用git merge合并在一起

git merge 0.2

（这个作用主要用于团队协作）

如：

项目主分支 1.0版本

1.1版本需要添加5个新功能，我把这个5个功能分配给5个小伙伴，他们就在1.0版本上branch出去一个自己的版本，分支就命名为1.0-功能xxx

然后5个人同时开发，到时间后，我在主分支上，把他们5个人的5个分支都merge过来，再提交并使用git tag（tag命令创建固定版本）（打标签1.1)

---

团队协作肯定不在同一台电脑上，这时候需要一台服务器搭建一个git仓库服务（GitLab、Bitbucket），自己搭建也不难，有gitlab之类的开源库可以做企业私有服务，但是大部分人没这个条件或者就是懒得搭建，那么就可以使用github或者gitee这种公共的git仓库。说白了他们就是个符合git操作的网盘。

---
### 真正开始

如何自己在github建一个库？

在github上，new repository，之后有教程。

git init
git add README.md
git commit -m "first commit"  这三个都已学过做过

git branch -M main (创建一个main分支，并把主分支切换为main)

git remote add origin https://github.com/himenotowa/xxx.git （仓库网址+.git）（相当于给这个git项目设置一个网盘地址，这样git就知道该上传到哪里了）然后会提示输入邮箱和密码，输入github的即可。

git pull origin main           // 刚创建一个新库，所以要把东西先从远端下载过来，保证一样。

(有可能出现错误，这是因为文件版本没有及时更新，两个分支是两个不同的版本，具有不同的提交历史，解决方式就是在原本的命令之后加上一句命令即可：
git pull origin main --allow-unrelated-histories)

git push -u origin main （push就是推送上传到github）(-u即--set-upstream)


---

如何参与开发开源项目？

仓库为public即可参与

https://github.com/midorg-com/re01

点击右上角的Fork （相当于把他（元岛）的代码库复制到自己的账号里，类似于branch）

然后右上角回到自己的仓库，可以看到已经复制过来了，这就是他的项目在自己账号的branch。

点击绿色的code按钮，复制一下仓库的https....git链接

然后，在自己电脑上找个文件夹，用vscode打开，新建终端，输入
git clone 仓库链接 空格点号. （这个命令叫克隆，把网盘上的仓库克隆到本地电脑）（失败的话多试几次，网络问题）

git remote -v （可以看到，只有自己仓库的链接）

所以再去元岛的仓库，复制一下项目的链接，回到本地

git remote add upstream 链接  （添加上游代码库）

git remote -v (这次可以正常看到上游链接)

如果要给别人加功能，可以先创建一个分支

git checkout -b kwc （创建并切换进入kwc的分支）

创建members文件夹

文件夹里创建kwc.json，内容为：

{

​	"name": "kwc"

​	"uri": "https://kwc.com/"

}

保存

git add .

git commit -m "add(member): kwc"

git push

git push -u origin kwc

回到github网页刷新，可以看到自己有这个分支了。

这时候，来到re01（元岛）的仓库，点击Pull requests（拉取请求），简称PR

进去后点绿色的New pull request

点compare across forks

选择根库和compare:自己的branch

如果显示绿色√，则可以合并，则创建PR，填写标题内容，点右下角创建PR就提交成功了。这时主分支的人就可以去合并操作了。

如果没显示绿色√，那可能是写代码时，主分支的人提交了新的commit，导致版本不一致，这时候需要先更新一下本地版本：

git fetch upstream

git merge upstream/main

git push

这时候快点去提交pr，就可以了。





---



