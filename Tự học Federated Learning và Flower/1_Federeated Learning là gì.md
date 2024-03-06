# Federated Learning
## 1. Classic Machine Learning
Trong Machine Learning, chúng ta có một mô hình, và dữ liệu. Chúng ta sẽ huấn luyện mô hình trên dữ liệu nhằm thực hiện một tác vụ nào đó.

Trong thực tế, dữ liệu dùng để huấn luyện thường không đến từ máy mà chúng ta thực hiện việc huấn luyện mà đến từ những nguồn khác. Do vậy dữ liệu sẽ được thu thập và đưa đến một nơi duy nhất, thường là một máy chủ đám mây.

## 2. Thách thức của machine Learning
Phương pháp huấn luyện Machine Leaning tập chung thường gặp phải một số những vấn đề như sau:
- Các điều luật ngăn cản việc thu thập và gửi dữ liệu đi như: GDPR (Europe), CCPA (California), PIPEDA (Canada), LGPD (Brazil), PDPL (Argentina), KVKK (Turkey), POPI (South Africa), FSS (Russia), CDPR (China), PDPB (India), PIPA (Korea), APPI (Japan), PDP (Indonesia), PDPA (Singapore), APP (Australia), etc.
- Người dùng không cho phép việc thu thập và lấy dữ liệu cá nhân
- Lượng dữ liệu là quá lớn và không thể truyền tải cũng như lưu trữ hết, trong đó có cả những dữ liệu không có ích.

Một số trường hợp mà phương pháp học tập chung không thể được thực hiện như:
- Dữ liệu sức khỏe từ các bệnh viện nhằm huấn luyện mô hình phát hiện bệnh
- Dữ liệu tài chính của các tổ chức nhằm phát hiện lừa đảo tài chính
- Dữ liệu cá nhân của người dùng
- Dữ liệu được mã hóa

## 3. Federated Learning
Federated Learning giải quyết các vấn đề của học máy tập chung bằng cách cho phép áp dụng học máy lên dữ liệu phân tán. Thay vì đưa dữ liệu đến mô hình thì FL đưa mô hình đến dữ liệu.

Federated Learning hoạt đông bằng 5 bước sau:
- Bước 0: Khởi tạo mô hình toàn cầu, tương tự như mô hình truyền thống.
- Bước 1: Gửi mô hình đến một tập các máy khách (clients)
- Bước 2: Huấn luyện mô hình một cách cục bộ, mô hình không nhất thiết phải huấn luyện đến khi hội tụ.
- Bước 3: Trả mô hình về với máy chủ
- Bước 4: Tổng hợp các mô hình thành một mô hình toàn cầu mới
- Bước 5: Lặp lại từ bước 1-4 cho đến khi mô hình hội tụ
