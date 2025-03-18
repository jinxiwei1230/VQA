data = {
    'data': '[{"bg":"0","ed":"600","onebest":"嗯","speaker":"0"},'
            '{"bg":"870","ed":"2640","onebest":"嗯","speaker":"0"},'
            '{"bg":"2640","ed":"6880","onebest":"嗯","speaker":"0"},'
            '{"bg":"6880","ed":"7680","onebest":"嗯","speaker":"0"},'
            '{"bg":"7720","ed":"11540","onebest":"嗯","speaker":"0"},'
            '{"bg":"12210","ed":"14320","onebest":"嗯","speaker":"0"},'
            '{"bg":"14320","ed":"16700","onebest":"外面太危险了，","speaker":"0"},'
            '{"bg":"17000","ed":"20020","onebest":"有人在追杀我，你不能出去，嗯","speaker":"0"},'
            '{"bg":"20250","ed":"21180","onebest":"唉","speaker":"0"},'
            '{"bg":"21230","ed":"23570","onebest":"不好意思。","speaker":"0"},'
            '{"bg":"23570","ed":"26940","onebest":"我刚才拽你的时候伤口撑开了，","speaker":"0"},'
            '{"bg":"27410","ed":"28400","onebest":"嗯","speaker":"0"},'
            '{"bg":"32950","ed":"35630","onebest":"妈呀这是中了多少枪？啊","speaker":"0"},'
            '{"bg":"35670","ed":"37070","onebest":"草船借箭。哪","speaker":"0"},'
            '{"bg":"37430","ed":"40900","onebest":"快帮我找一下，真和谐，我要缝合。","speaker":"0"},'
            '{"bg":"43260","ed":"44280","onebest":"那什么？","speaker":"0"},'
            '{"bg":"44280","ed":"46280","onebest":"你这点小病我就能治。","speaker":"0"},'
            '{"bg":"46570","ed":"49860","onebest":"嗯嗯啊啊啊对哦好了。","speaker":"0"},'
            '{"bg":"49860","ed":"50660","onebest":"记得给我写信。","speaker":"0"},'
            '{"bg":"50660","ed":"52240","onebest":"呢","speaker":"0"},'
            '{"bg":"53000","ed":"65670","onebest":"太好了，老师老师龚蓓蓓导演这样不合理，啊我后面还有展示疼痛的舞蹈你拉倒，吧事实证明了这种拍摄方式确实不适合你位老师，别别我我们也是设计和你就好像是牛郎与织女","speaker":"0"},'
            '{"bg":"65670","ed":"69150","onebest":"永远不能在一起，这是怎么了？","speaker":"0"},'
            '{"bg":"69150","ed":"70050","onebest":"看着没有？","speaker":"0"},'
            '{"bg":"70050","ed":"71310","onebest":"先迷糊一个。","speaker":"0"},'
            '{"bg":"93150","ed":"94120","onebest":"嗯","speaker":"0"},'
            '{"bg":"95260","ed":"96400","onebest":"谁？啊","speaker":"0"},'
            '{"bg":"96560","ed":"98520","onebest":"是我米兰老师。","speaker":"0"},{"bg":"103260","ed":"106750","onebest":"唉导演魏老师找我就有事吗？","speaker":"0"},{"bg":"106770","ed":"108220","onebest":"啊嗯","speaker":"0"},{"bg":"108520","ed":"116710","onebest":"唉是这样有点冒昧，啊我父母今天来探班来了，所以想请米兰老师我们一起过去吃个饭。","speaker":"0"},{"bg":"116710","ed":"118750","onebest":"您看您您看小了李老师。","speaker":"0"},{"bg":"118970","ed":"120510","onebest":"我姐今天不太方便。","speaker":"0"},{"bg":"120510","ed":"121380","onebest":"要不","speaker":"0"},{"bg":"121380","ed":"122100","onebest":"感恩，","speaker":"0"},{"bg":"123030","ed":"125560","onebest":"好好好好","speaker":"0"},{"bg":"130120","ed":"130980","onebest":" enough。","speaker":"0"},{"bg":"131780","ed":"133210","onebest":"咱们还剩多少时间？","speaker":"0"},{"bg":"133300","ed":"134410","onebest":"三个小时。","speaker":"0"},{"bg":"134990","ed":"135830","onebest":"嗯","speaker":"0"},{"bg":"140330","ed":"141990","onebest":"我觉得我应该过去一下，","speaker":"0"},{"bg":"142130","ed":"143310","onebest":"把真相告诉他。","speaker":"0"},{"bg":"143600","ed":"144840","onebest":"别呀姐，","speaker":"0"},{"bg":"144840","ed":"150400","onebest":"他要知道这项一胡闹咱们还走得了吗？这艘渔船我都费了好大劲才找来，再不走就真得等下周了。","speaker":"0"},{"bg":"150560","ed":"151520","onebest":"米勒","speaker":"0"},{"bg":"151590","ed":"153090","onebest":"从小我就告诉你，","speaker":"0"},{"bg":"153090","ed":"154350","onebest":"做人要负责任，","speaker":"0"},{"bg":"154510","ed":"156350","onebest":"这事和魏成功有什么关系？","speaker":"0"},{"bg":"156350","ed":"157920","onebest":"咱不能就这么一走了之了。","speaker":"0"},{"bg":"159120","ed":"159960","onebest":"这回","speaker":"0"},{"bg":"160460","ed":"161970","onebest":"我要坦坦荡荡一次，","speaker":"0"},{"bg":"162350","ed":"163360","onebest":"嗯","speaker":"0"},{"bg":"176190","ed":"177710","onebest":"米兰老师。","speaker":"0"}]',
    'err_no': 0, 'failed': None, 'ok': 0}
print(type(data))
subtitle_path = vs_common.hdfs_result_store_path.format(id) + "/subtitle/"
if not os.path.exists(subtitle_path):
        self.hdfs_client.makedirs(subtitle_path, 777)
self.hdfs_client.write(subtitle_path + "asr_result.json", json.dumps(result), overwrite=True)
