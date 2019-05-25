menu = {
    '北京': {
        '朝阳': {
            'CICC': {},
            'HP': {},
            "渣打银行": {},
            'CCTV': {},
        },
        '望京': {
            '陌陌': {},
            '奔驰': {},
            '360': {},
        },
        '三里屯': {
            '优衣库': {},
            'apple': {},
        },
        '昌平': {
            '沙河': {
                '老男孩': {},
                '阿泰包子': {},
            },
            '天通苑': {
                '链家': {},
                '我爱我家': {},
            },
            '回龙观': {},
            },
        '海淀': {
            '五道口':{
                '谷歌': {},
                '网易': {},
                'sohu': {},
                'sogo': {},
                '快手': {},
            },
            '中关村': {
            'youku': {},
            'Iqiyi': {},
            'QQ': {},
            },
        },
    },
    '武汉': {
        "武昌": {},
        "汉口": {},
        "汉阳": {},
    },
    '广州': {
        '白云': {},
        '佛山': {},
        '肇庆': {},
    },
}

current_layer = menu
parent_layers = []
while True:
    for key in current_layer:
        print(key)
    choice = input(">>>:").strip()
    if len(choice) == 0:
        continue
    if choice in current_layer:
        parent_layers.append(current_layer)
        current_layer = current_layer[choice]
    elif choice == "b":
        if parent_layers:
            current_layer = parent_layers.pop()
    else:print("无此项")



