import scapy.all as scapy
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def create_dataset():
    path = r'D:\WiresharkPortable64\App\Wireshark\tshark.exe'
    net = 'trace_2023_01_29_09_37_10.pcap5'
    df = capture_data(net)
    df.to_csv('ddos.csv')


def capture_data(files):
    pkts = scapy.rdpcap(files)  # Membaca file PCAP

    data = []
    start_time = pkts[0].time
    end_time = pkts[-1].time
    dur = end_time - start_time

    for pkt in pkts:
        if pkt.haslayer(scapy.IP):
            ip_src = pkt[scapy.IP].src
            ip_dst = pkt[scapy.IP].dst

            data.append({
                "ip src": ip_src,
                "ip dst": ip_dst,
                "pktcount": 1,  # Setiap paket dihitung sebagai 1
                "bytecount": len(pkt),
                "dur": dur,
            })

    df = pd.DataFrame(data)
    grouped = df.groupby(["ip src", "ip dst"]).agg({
        "pktcount": "sum",
        "bytecount": "sum",
        "dur": "max"
    }).reset_index()

    grouped["pktrate"] = grouped["pktcount"] / grouped["dur"]

    threshold = 100  # Misalnya, jika lebih dari 100 paket dalam durasi tertentu, itu dianggap sebagai serangan DDoS
    grouped["label"] = grouped["pktcount"].apply(lambda x: "Serangan DDoS" if x > threshold else "Tidak ada serangan DDoS")
    return grouped


if __name__ == '__main__':
    # create_dataset()
    df = pd.read_csv('ddos.csv')

    x = df[['pktcount', 'bytecount', 'dur', 'pktrate']]
    y = df['label']

    pd.set_option('display.max_columns', None)
    print(df.head(5))
    print(df.tail(5))

    encode = LabelEncoder()
    y = encode.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    svm = SVC(kernel='linear', C=1)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy)



