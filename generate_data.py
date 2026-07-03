import csv
import random
import re

ACCENT_MAP = {
    'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
    'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
    'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
    'đ': 'd',
    'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
    'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
    'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
    'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
    'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
    'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
    'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
    'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
    'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
    'Đ': 'D'
}

def remove_accents(text):
    res = []
    for char in text:
        res.append(ACCENT_MAP.get(char, char.lower() if char.isupper() and char.lower() in ACCENT_MAP else char))
    return "".join(res)

# Richer typo generator targeting medical vocabulary
def make_typos(text):
    words = text.split()
    new_words = []
    for w in words:
        # High chance to swap typical Vietnamese digraphs
        if random.random() < 0.15:
            w = w.replace("ch", "tr").replace("tr", "ch")
            w = w.replace("s", "x").replace("x", "s")
            w = w.replace("d", "gi").replace("gi", "d").replace("r", "d")
            w = w.replace("ng", "n").replace("nh", "n")
            w = w.replace("iê", "yê").replace("yê", "iê")
            w = w.replace("uô", "uo").replace("ươ", "uo")
        # Random typing typo
        if random.random() < 0.1:
            if len(w) > 3:
                idx = random.randint(1, len(w)-2)
                w = w[:idx] + w[idx+1] + w[idx] + w[idx+2:]
        new_words.append(w)
    return " ".join(new_words)

# Predefined Fillers
FILLERS_PREFIX = [
    "", "bác sĩ ơi cho em hỏi", "dạ bác sĩ cho em hỏi", "bác sĩ ơi", 
    "bác sĩ cho em hỏi chút ạ", "cho em hỏi", "cho hỏi", "dạ cho em hỏi", 
    "dạ thưa bác sĩ", "alo bác sĩ", "bác sĩ giải đáp giúp em", 
    "bác sĩ tư vấn giúp em", "em muốn hỏi là", "mọi người cho em hỏi", 
    "cho em hỏi tí ạ", "dạ bác sĩ", "thưa bác sĩ", "bác sĩ ơi giúp em với", 
    "em chào bác sĩ ạ", "chào bác sĩ ạ"
]

FILLERS_SUFFIX = [
    "", "ạ", "thế ạ", "nha bác sĩ", "ạ em cảm ơn bác sĩ", "ạ cảm ơn bác sĩ", 
    "ạ xin cảm ơn", "với ạ", "giúp em với ạ", "ạ?", "ạ !", "nhe bác sĩ", 
    "giúp em", "xin cảm ơn bác sĩ", "nha", "nhé bác sĩ", "ạ em cảm ơn", 
    "ạ xin cảm ơn nhiều"
]

def apply_fillers(sentence, p=None, s=None):
    if p is None:
        p = random.choice(FILLERS_PREFIX)
    if s is None:
        s = random.choice(FILLERS_SUFFIX)
    
    parts = []
    if p:
        parts.append(p)
    parts.append(sentence)
    if s:
        parts.append(s)
        
    res = " ".join(parts)
    res = res.strip()
    if res:
        res = res[0].upper() + res[1:]
    return res

# Medical Entities Dictionary for standardization & rich queries
MEDICAL_ENTITIES = {
    "paracetamol": {
        "abbrs": ["para", "paxetamol", "panadol", "pcm", "pana"],
        "synonyms": ["thuốc hạ sốt paracetamol", "thuốc giảm đau para"]
    },
    "amoxicillin": {
        "abbrs": ["amox", "amoxi", "amoc", "klamentin"],
        "synonyms": ["kháng sinh amoxicillin", "thuốc amox"]
    },
    "ibuprofen": {
        "abbrs": ["ibu", "ibup", "ibuf"],
        "synonyms": ["thuốc kháng viêm ibuprofen", "giảm đau kháng viêm ibu"]
    },
    "aspirin": {
        "abbrs": ["aspi", "asprin"],
        "synonyms": ["thuốc tim mạch aspirin", "asprin chống đông máu"]
    },
    "metformin": {
        "abbrs": ["met", "metfor"],
        "synonyms": ["thuốc tiểu đường metformin", "metformin hạ đường huyết"]
    },
    "tăng huyết áp": {
        "abbrs": ["ha cao", "huyet ap cao", "ha"],
        "synonyms": ["cao huyết áp", "huyết áp cao", "bệnh cao huyết áp"]
    },
    "viêm loét dạ dày": {
        "abbrs": ["dau bao tu", "loet da day"],
        "synonyms": ["đau dạ dày", "đau bao tử", "viêm loét bao tử"]
    },
    "đái tháo đường": {
        "abbrs": ["tieu duong", "dtd", "duong huyet cao"],
        "synonyms": ["bệnh tiểu đường", "tiểu đường tuýp 2", "đường huyết cao"]
    },
    "rối loạn lipid máu": {
        "abbrs": ["mo mau", "mo mau cao", "cholesterol cao"],
        "synonyms": ["mỡ máu cao", "cholesterol cao", "mỡ máu"]
    },
    "gout": {
        "abbrs": ["gut", "benh gut"],
        "synonyms": ["bệnh gout", "sưng khớp do gút", "cơn gút cấp"]
    },
    "hen phế quản": {
        "abbrs": ["hen", "suen", "kho khe"],
        "synonyms": ["bệnh hen suyễn", "hen suyễn", "co thắt phế quản"]
    },
    "rối loạn tiền đình": {
        "abbrs": ["tien dinh", "roi loan td"],
        "synonyms": ["đau đầu tiền đình", "chóng mặt tiền đình", "xây xẩm tiền đình"]
    }
}

# Generate hard cases (medical vs emergency boundaries)
def generate_hard_cases():
    # Pairs of (medical_opt, emergency_opt)
    hard_templates = [
        # Cardiovascular/Chest Pain
        (
            "Tôi hay bị đau nhói nhẹ lồng ngực vùng tim khi hít thở sâu hoặc thay đổi tư thế",
            "Cấp cứu gấp người nhà đột ngột đau ngực dữ dội như đá đè lan ra vai trái và vã mồ hôi lạnh ngắt"
        ),
        (
            "Cảm giác đau tức ngực âm ỉ nhẹ kéo dài vài tuần nay có cần đi khám tim mạch không",
            "Cấp cứu! Cơn đau thắt bóp nghẹt lồng ngực lan ra hàm và sau lưng kèm khó thở dữ dội"
        ),
        (
            "Thỉnh thoảng bị nặng ngực sau khi đi bộ nhanh hoặc leo cầu thang cao",
            "Người bệnh đột ngột tức ngực ồ ạt, tay chân lạnh toát và ngất lịm đi không tỉnh táo"
        ),
        # Neurological/Headache/Stroke
        (
            "Đau nhức đầu âm ỉ vùng sau gáy sau khi thức đêm làm việc mệt mỏi",
            "Người nhà đột ngột đau đầu dữ dội như búa bổ chưa từng có kèm theo buồn nôn liên tục"
        ),
        (
            "Bị đau nửa đầu kinh niên khi thay đổi thời tiết chuyển lạnh đột ngột",
            "Khẩn cấp! Người bệnh đột ngột méo miệng, liệt hoàn toàn một bên người và không thể nói rõ tiếng"
        ),
        (
            "Thỉnh thoảng bị tê bì đầu ngón tay nhẹ hoặc châm chích ở cánh tay",
            "Người nhà đột ngột bị té ngã bất tỉnh, co giật toàn thân trợn mắt sùi bọt mép"
        ),
        # Respiratory/Airway
        (
            "Bị ho khan khò khè nhẹ kèm theo ngứa họng do dị ứng bụi bẩn",
            "Khẩn cấp! Trẻ bị hóc hạt nhãn dị vật đường thở tắc nghẽn hoàn toàn đang tím tái môi không thở được"
        ),
        (
            "Thở khò khè nhẹ có đờm ở cổ họng sau khi ngủ dậy",
            "Cấp cứu gấp người bệnh ngừng tuần hoàn, ngừng thở hoàn toàn lồng ngực không phập phồng"
        ),
        # Trauma/Bleeding
        (
            "Bị xước da đầu gối rớm ít máu sau khi ngã xe nhẹ ngoài đường",
            "Người nhà bị tai nạn gãy xương đùi đâm thủng da lồi hẳn ra ngoài máu chảy ồ ạt không ngừng"
        ),
        (
            "Bị bong gân nhẹ cổ chân gây sưng tấy đau nhức khi vận động mạnh",
            "Cấp cứu tai nạn lao động bị đứt động mạch cánh tay máu phun thành tia xối xả làm sao cầm máu khẩn cấp"
        ),
        (
            "Vết đứt tay nhỏ do dao gọt hoa quả chảy ít máu đã được rửa sạch",
            "Người nhà bị ngã từ tầng cao xuống chấn thương sọ não máu chảy nhiều ở đầu và tai mũi"
        ),
        # Pediatric/Fever/Seizure
        (
            "Bé bị sốt nhẹ 38 độ mệt mỏi nhưng vẫn bú sữa mẹ bình thường",
            "Bé sốt cao trên 39.5 độ đột ngột co giật liên hồi trợn mắt tím tái môi"
        ),
        (
            "Trẻ bị sốt hâm hấp nóng người sau khi đi tiêm phòng về",
            "Trẻ sơ sinh đột ngột ngưng thở hoàn toàn toàn thân lạnh ngắt sau khi bú"
        )
    ]

    med_records = []
    emerg_records = []
    
    seen_med = set()
    seen_emerg = set()
    
    # We want ~1500 medical and ~1500 emergency hard cases
    for m_tpl, e_tpl in hard_templates:
        # Generate variations for each template
        for _ in range(150):
            # 1. Medical side
            p = random.choice(FILLERS_PREFIX)
            s = random.choice(FILLERS_SUFFIX)
            m_opt = apply_fillers(m_tpl, p, s)
            
            if m_opt not in seen_med:
                seen_med.add(m_opt)
                m_raw = m_opt
                if random.random() < 0.4:
                    m_raw = make_typos(m_raw)
                if random.random() < 0.5:
                    m_raw = remove_accents(m_raw)
                m_raw = re.sub(r'\s+', ' ', m_raw).strip()
                m_opt = re.sub(r'\s+', ' ', m_opt).strip()
                med_records.append((m_raw, m_opt, "medical"))
                
            # 2. Emergency side
            p = random.choice(FILLERS_PREFIX)
            s = random.choice(FILLERS_SUFFIX)
            e_opt = apply_fillers(e_tpl, p, s)
            
            if e_opt not in seen_emerg:
                seen_emerg.add(e_opt)
                e_raw = e_opt
                if random.random() < 0.4:
                    e_raw = make_typos(e_raw)
                if random.random() < 0.5:
                    e_raw = remove_accents(e_raw)
                e_raw = re.sub(r'\s+', ' ', e_raw).strip()
                e_opt = re.sub(r'\s+', ' ', e_opt).strip()
                emerg_records.append((e_raw, e_opt, "emergency"))
                
    return med_records, emerg_records

# Generate MEDICAL category using entity mappings for entity standardization
def generate_medical(limit=10000):
    templates_base = [
        "thuốc {drug} được chỉ định để điều trị {symptom}",
        "tác dụng phụ của {drug} khi dùng chung với {other_drug} là gì",
        "liều lượng uống {drug} cho người bị {symptom} như thế nào",
        "tôi bị {symptom} thì có nên uống {drug} không",
        "cơ chế tác động của {drug} trong điều trị {disease} là gì",
        "dùng {drug} kéo dài có gây suy gan hay suy thận không",
        "phụ nữ có thai có được sử dụng {drug} để chữa {symptom}",
        "tôi uống {drug} quá liều thì nên xử lý thế nào",
        "thuốc {drug} có tương tác với các loại thuốc điều trị {disease} không",
        "lưu ý quan trọng nhất khi sử dụng {drug} là gì",
        "tại sao dùng {drug} lại gây buồn ngủ hoặc chóng mặt",
        "bệnh nhân {disease} có cần kiêng khem gì khi uống {drug} không",
        "sự khác biệt giữa {drug} và {other_drug} trong chữa trị {symptom}",
        "thời điểm uống {drug} tốt nhất trong ngày là lúc nào",
        "trẻ em dưới 6 tuổi uống {drug} liều bao nhiêu là an toàn",
    ]

    # Map the standardized terms to placeholders
    # Standard drugs
    standard_drugs = ["paracetamol", "amoxicillin", "ibuprofen", "aspirin", "metformin"]
    # Standard diseases/symptoms
    standard_diseases = ["tăng huyết áp", "viêm loét dạ dày", "đái tháo đường", "rối loạn lipid máu", "gout", "hen phế quản", "rối loạn tiền đình"]
    standard_symptoms = ["đau đầu", "đau bụng", "sốt", "chóng mặt", "mất ngủ"]

    records = []
    seen = set()
    
    pool = []
    for tpl in templates_base:
        # generate plenty of candidates
        for _ in range(1200):
            d1 = random.choice(standard_drugs)
            d2 = random.choice([d for d in standard_drugs if d != d1])
            dis = random.choice(standard_diseases)
            sym = random.choice(standard_symptoms)
            
            # Form clean/optimized query using standardized terms
            opt_sentence = tpl.format(drug=d1, other_drug=d2, symptom=sym, disease=dis)
            pool.append((opt_sentence, d1, d2, dis, sym))

    random.shuffle(pool)
    
    for opt_sentence, d1, d2, dis, sym in pool:
        if len(records) >= limit:
            break
            
        p = random.choice(FILLERS_PREFIX)
        s = random.choice(FILLERS_SUFFIX)
        opt = apply_fillers(opt_sentence, p, s)
        
        if opt in seen:
            continue
        seen.add(opt)
        
        # Build the raw query with abbreviations/synonyms to teach normalization
        raw = opt
        
        # Replace drug 1
        d1_entity = MEDICAL_ENTITIES[d1]
        d1_replace = random.choice(d1_entity["abbrs"] + d1_entity["synonyms"] + [d1])
        raw = raw.replace(d1, d1_replace)
        raw = raw.replace(d1.capitalize(), d1_replace.capitalize())
        
        # Replace drug 2 if present
        if d2 in raw:
            d2_entity = MEDICAL_ENTITIES[d2]
            d2_replace = random.choice(d2_entity["abbrs"] + d2_entity["synonyms"] + [d2])
            raw = raw.replace(d2, d2_replace)
            raw = raw.replace(d2.capitalize(), d2_replace.capitalize())
            
        # Replace disease if present
        if dis in raw:
            dis_entity = MEDICAL_ENTITIES[dis]
            dis_replace = random.choice(dis_entity["abbrs"] + dis_entity["synonyms"] + [dis])
            raw = raw.replace(dis, dis_replace)
            raw = raw.replace(dis.capitalize(), dis_replace.capitalize())
            
        # Replace symptom if present
        if sym in raw:
            # check if it exists in MEDICAL_ENTITIES
            if sym in MEDICAL_ENTITIES:
                sym_entity = MEDICAL_ENTITIES[sym]
                sym_replace = random.choice(sym_entity["abbrs"] + sym_entity["synonyms"] + [sym])
            else:
                # fall back to standard list
                sym_replace = sym
            raw = raw.replace(sym, sym_replace)
            raw = raw.replace(sym.capitalize(), sym_replace.capitalize())

        # Typos
        if random.random() < 0.4:
            raw = make_typos(raw)
            
        # Strip accents
        if random.random() < 0.5:
            raw = remove_accents(raw)
            
        raw = re.sub(r'\s+', ' ', raw).strip()
        opt = re.sub(r'\s+', ' ', opt).strip()
        
        records.append((raw, opt, "medical"))
        
    return records

# Generate EMERGENCY category
def generate_emergency(limit=10000):
    templates = [
        "người nhà bị {condition} đang rất {urgency} cứu với",
        "bị {condition} kèm {symptom} nguy kịch quá, phải sơ cứu sao",
        "khẩn cấp bệnh nhân {condition} đang ngừng thở, có cần hô hấp nhân tạo không",
        "con uống nhầm {poison} sùi bọt mép, cứu mạng với",
        "tai nạn xe máy làm nạn nhân bị {condition} chảy máu ồ ạt, làm sao cầm máu",
        "bị đột quỵ {condition} méo miệng tê liệt nửa người gọi cấp cứu chưa thấy tới phải làm gì",
        "cấp cứu gấp người bị {condition} đột ngột ngất xỉu tím tái toàn thân",
        "bị {condition} nghẹt thở ho sặc sụa trợn mắt lên thì sơ cứu thế nào",
    ]

    conditions = [
        "đau ngực dữ dội như đá đè", "chấn thương sọ não máu chảy nhiều", "đuối nước",
        "bỏng nặng toàn thân do nổ bình gas", "gãy xương đùi đâm thủng da", "co giật trợn mắt sùi bọt mép",
        "hôn mê sâu không phản ứng", "vỡ mạch máu phun thành tia", "dị vật đường thở tắc hoàn toàn",
        "đột quỵ não liệt hoàn toàn nửa người", "phản vệ độ nặng sau tiêm thuốc", "nuốt nhầm acid"
    ]

    symptoms = [
        "tím tái môi và đầu ngón tay", "mạch đập cực kỳ yếu", "ngừng thở hoàn toàn",
        "co giật liên hồi", "mất nhận thức", "nôn ra máu tươi ồ ạt", "vã mồ hôi lạnh ngắt"
    ]

    poisons = [
        "thuốc diệt cỏ paraquat", "thuốc trừ sâu", "thuốc ngủ liều lượng cực lớn",
        "nước tẩy bồn cầu", "dầu hỏa", "hóa chất độc hại"
    ]

    urgencies = [
        "nguy kịch", "khẩn cấp", "nguy hiểm tính mạng", "gấp rút", "nguy cấp"
    ]

    records = []
    seen = set()
    
    pool = []
    for tpl in templates:
        for _ in range(2000):
            cond = random.choice(conditions)
            sym = random.choice(symptoms)
            poi = random.choice(poisons)
            urg = random.choice(urgencies)
            base_sentence = tpl.format(condition=cond, symptom=sym, poison=poi, urgency=urg)
            pool.append(base_sentence)
            
    random.shuffle(pool)
    
    for base_sentence in pool:
        if len(records) >= limit:
            break
            
        p = random.choice(FILLERS_PREFIX)
        s = random.choice(FILLERS_SUFFIX)
        opt = apply_fillers(base_sentence, p, s)
        
        if opt in seen:
            continue
        seen.add(opt)
        
        raw = opt
        if random.random() < 0.4:
            raw = make_typos(raw)
        if random.random() < 0.5:
            raw = remove_accents(raw)
            
        raw = re.sub(r'\s+', ' ', raw).strip()
        opt = re.sub(r'\s+', ' ', opt).strip()
        
        records.append((raw, opt, "emergency"))
        
    return records

# Generate FAQ category
def generate_faq(limit=10000):
    templates = [
        "làm thế nào để {goal} hiệu quả nhất tại nhà",
        "tác dụng của {item} đối với sức khỏe là gì",
        "chatbot có thể thay thế bác sĩ khám bệnh hoàn toàn được không",
        "làm sao để bảo vệ {body_part} khi làm việc văn phòng nhiều",
        "những thói quen tốt giúp nâng cao {health_aspect}",
        "lợi ích của việc {activity} vào mỗi buổi sáng/tối",
        "ăn gì hoặc kiêng gì để {goal} tốt nhất",
        "chatbot y tế này có những tính năng gì vậy",
    ]

    goals = [
        "giảm cân an toàn", "cải thiện giấc ngủ ngon", "tăng cường trí nhớ",
        "giảm stress căng thẳng", "giúp da dẻ mịn màng", "ổn định hệ tiêu hóa",
        "giữ dáng thon gọn", "thải độc cơ thể", "tăng cân lành mạnh", "hạ mỡ máu tự nhiên",
        "tăng cường thể lực", "giảm đau mỏi vai gáy", "chăm sóc da mụn", "tăng sự tập trung"
    ]

    items = [
        "trà hoa cúc", "uống nhiều nước lọc", "ăn rau xanh quả mọng", "tinh bột nghệ",
        "mật ong ấm", "tỏi đen", "sữa chua không đường", "nhân sâm", "nước ép cần tây",
        "trà xanh", "hạt chia", "dầu ô liu", "quế và gừng", "nước dừa"
    ]

    body_parts = [
        "mắt và cột sống cổ", "sức khỏe tim mạch", "hệ hô hấp", "làn da vùng mặt",
        "khớp xương cổ tay", "lá gan và quả thận", "hệ tiêu hóa", "xương khớp toàn thân",
        "não bộ và hệ thần kinh"
    ]

    health_aspects = [
        "hệ miễn dịch", "sức đề kháng mùa lạnh", "sức khỏe tinh thần",
        "độ dẻo dai xương khớp", "tuổi thọ cơ thể", "sức bền tim mạch", "khả năng tập trung",
        "chất lượng giấc ngủ"
    ]

    activities = [
        "chạy bộ 30 phút", "tập thiền và yoga", "uống một ly nước ấm",
        "đi bộ nhẹ nhàng", "giãn cơ thư giãn", "tập gym đều đặn", "ngủ đủ 8 tiếng",
        "hạn chế dùng điện thoại trước khi ngủ"
    ]

    records = []
    seen = set()
    
    pool = []
    for tpl in templates:
        for _ in range(2500):
            goal = random.choice(goals)
            itm = random.choice(items)
            bp = random.choice(body_parts)
            ha = random.choice(health_aspects)
            act = random.choice(activities)
            base_sentence = tpl.format(goal=goal, item=itm, body_part=bp, health_aspect=ha, activity=act)
            pool.append(base_sentence)
            
    random.shuffle(pool)
    
    for base_sentence in pool:
        if len(records) >= limit:
            break
            
        p = random.choice(FILLERS_PREFIX)
        s = random.choice(FILLERS_SUFFIX)
        opt = apply_fillers(base_sentence, p, s)
        
        if opt in seen:
            continue
        seen.add(opt)
        
        raw = opt
        if random.random() < 0.4:
            raw = make_typos(raw)
        if random.random() < 0.5:
            raw = remove_accents(raw)
            
        raw = re.sub(r'\s+', ' ', raw).strip()
        opt = re.sub(r'\s+', ' ', opt).strip()
        
        records.append((raw, opt, "faq"))
        
    return records

# Generate OUT-OF-SCOPE category
def generate_oos(limit=10000):
    templates = [
        "làm thế nào để {action} bằng {tech}",
        "chỉ giúp tôi công thức {cook} ngon chuẩn vị với",
        "cách {action} đẹp thu hút nhiều lượt tương tác",
        "tìm hiểu về {topic} trong lịch sử/khoa học",
        "dự báo thời tiết hôm nay tại {location} thế nào",
        "hướng dẫn các bước {fix} tại nhà chi tiết",
        "tôi muốn đầu tư vào {finance} thì nên bắt đầu từ đâu",
        "viết hộ tôi một bài thơ về {poet_topic} hay và ý nghĩa",
    ]

    actions = [
        "lập trình web", "chụp ảnh phong cảnh", "thiết kế đồ họa",
        "viết code python", "học tiếng Anh giao tiếp", "tạo slide thuyết trình",
        "chỉnh sửa video ngắn"
    ]

    techs = [
        "React và Nodejs", "Photoshop", "AI Generative tools",
        "Python FastAPI", "Canva", "Premiere Pro"
    ]

    cooks = [
        "nấu phở bò Hà Nội", "làm bánh mì Việt Nam", "nấu lẩu thái chua cay",
        "kho cá trắm đưa cơm", "làm kim chi Hàn Quốc"
    ]

    topics = [
        "cuộc chiến tranh thế giới thứ hai", "thuyết tương đối của Einstein",
        "sự phát triển của AI thế kỷ 21", "hệ mặt trời và các hành tinh",
        "khảo cổ học Ai Cập cổ đại"
    ]

    locations = [
        "Hà Nội", "Thành phố Hồ Chí Minh", "Đà Nẵng", "Đà Lạt", "Nha Trang"
    ]

    fixes = [
        "sửa xe máy bị ngập nước", "vệ sinh điều hòa", "sửa vòi nước bị rò rỉ",
        "cài đặt lại hệ điều hành Windows"
    ]

    finances = [
        "thị trường chứng khoán", "bất động sản ven đô", "vàng vật chất",
        "quỹ mở tài chính"
    ]

    poet_topics = [
        "tình yêu đôi lứa", "quê hương đất nước", "mùa thu lá rụng",
        "tình cảm gia đình cha mẹ"
    ]

    records = []
    seen = set()
    
    pool = []
    for tpl in templates:
        for _ in range(2000):
            act = random.choice(actions)
            tch = random.choice(techs)
            ck = random.choice(cooks)
            tp = random.choice(topics)
            loc = random.choice(locations)
            fx = random.choice(fixes)
            fin = random.choice(finances)
            pt = random.choice(poet_topics)
            base_sentence = tpl.format(action=act, tech=tch, cook=ck, topic=tp, location=loc, fix=fx, finance=fin, poet_topic=pt)
            pool.append(base_sentence)
            
    random.shuffle(pool)
    
    for base_sentence in pool:
        if len(records) >= limit:
            break
            
        p = random.choice(FILLERS_PREFIX)
        s = random.choice(FILLERS_SUFFIX)
        opt = apply_fillers(base_sentence, p, s)
        
        if opt in seen:
            continue
        seen.add(opt)
        
        raw = opt
        if random.random() < 0.4:
            raw = make_typos(raw)
        if random.random() < 0.5:
            raw = remove_accents(raw)
            
        raw = re.sub(r'\s+', ' ', raw).strip()
        opt = re.sub(r'\s+', ' ', opt).strip()
        
        records.append((raw, opt, "out-of-scope"))
        
    return records

def main():
    print("Generating data...")
    
    # 1. Generate hard cases first (boundary cases)
    med_hard, emerg_hard = generate_hard_cases()
    print(f"Generated {len(med_hard)} medical hard cases.")
    print(f"Generated {len(emerg_hard)} emergency hard cases.")
    
    # Calculate how many standard cases we need to fill the rest of the 10,000 limit
    med_need = max(0, 10000 - len(med_hard))
    emerg_need = max(0, 10000 - len(emerg_hard))
    
    # 2. Generate standard cases
    med_std = generate_medical(limit=med_need)
    emerg_std = generate_emergency(limit=emerg_need)
    
    print(f"Generated {len(med_std)} standard medical records.")
    print(f"Generated {len(emerg_std)} standard emergency records.")
    
    # Combine standard and hard cases
    med = med_hard + med_std
    emerg = emerg_hard + emerg_std
    
    faq = generate_faq(limit=10000)
    print(f"Generated {len(faq)} faq records.")
    oos = generate_oos(limit=10000)
    print(f"Generated {len(oos)} out-of-scope records.")
    
    # Shuffle and trim to exactly 10,000 for balanced dataset
    random.shuffle(med)
    random.shuffle(emerg)
    random.shuffle(faq)
    random.shuffle(oos)
    
    med = med[:10000]
    emerg = emerg[:10000]
    faq = faq[:10000]
    oos = oos[:10000]
    
    # Save intent.csv
    intent_data = []
    intent_data.extend(med)
    intent_data.extend(emerg)
    intent_data.extend(faq)
    intent_data.extend(oos)
    
    random.shuffle(intent_data)
    
    with open('notebooks/intent.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        for raw, opt, label in intent_data:
            writer.writerow([raw, label])
            
    print(f"Saved {len(intent_data)} records to notebooks/intent.csv")
    
    # Save rewrite.csv (raw_query -> optimized_query)
    rewrite_data = []
    
    # Gather pairs where raw query != optimized query from all medical & emergency records
    # This includes the hard cases and medical entities standardizations!
    for raw, opt, label in med:
        if raw != opt:
            rewrite_data.append((raw, opt))
    for raw, opt, label in emerg:
        if raw != opt:
            rewrite_data.append((raw, opt))
    for raw, opt, label in faq:
        if raw != opt:
            rewrite_data.append((raw, opt))
            
    # Add identity mapping (opt -> opt) to teach model when NOT to rewrite
    # Grab 500 clean samples from medical and 500 from emergency
    exact_matches = 0
    for _, opt, _ in med[:500]:
        rewrite_data.append((opt, opt))
        exact_matches += 1
    for _, opt, _ in emerg[:500]:
        rewrite_data.append((opt, opt))
        exact_matches += 1
        
    random.shuffle(rewrite_data)
    
    with open('notebooks/rewrite.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['raw_query', 'optimized_query'])
        for raw, opt in rewrite_data:
            writer.writerow([raw, opt])
            
    print(f"Saved {len(rewrite_data)} records to notebooks/rewrite.csv (including {exact_matches} identity mapping records)")

if __name__ == '__main__':
    main()
