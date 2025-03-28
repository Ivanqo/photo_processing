<h1 align="center">✨ Smart Portrait Enhancer ✨</h1>

<p align="center">
  <strong>Автоматическая обработка портретов с AI-удалением фона и интеллектуальной коррекцией освещения</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python version">
  <img src="https://img.shields.io/badge/OpenCV-4.5+-green?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/Pillow-9.0+-yellow" alt="Pillow">
</p>

<div align="center">
  <img src="https://github.com/yourusername/smart-portrait-enhancer/raw/main/docs/demo.gif" width="600" alt="Демонстрация работы">
</div>

<h2>🚀 Возможности</h2>

<ul>
  <li>🔍 <strong>Автоматическое определение лиц</strong> - анализ яркости по лицу, а не по всему изображению</li>
  <li>💡 <strong>Интеллектуальная коррекция освещения</strong> - разные алгоритмы для темных, нормальных и пересвеченных фото</li>
  <li>🧼 <strong>Удаление фона</strong> с помощью нейросетевой модели</li>
  <li>📁 <strong>Пакетная обработка</strong> - поддержка массового преобразования фотографий</li>
  <li>🖼️ <strong>Сохранение в PNG</strong> с прозрачным фоном</li>
</ul>

<h2>🛠️ Технологии</h2>

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" height="28">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" height="28">
  <img src="https://img.shields.io/badge/Pillow-8F00FF?style=for-the-badge" height="28">
  <img src="https://img.shields.io/badge/Rembg-00A98F?style=for-the-badge" height="28">
</div>

<h2>📦 Установка</h2>

<pre><code># Клонировать репозиторий
git clone https://github.com/yourusername/smart-portrait-enhancer.git
cd smart-portrait-enhancer

# Установить зависимости
pip install -r requirements.txt
</code></pre>

<h2>🖥️ Использование</h2>

<pre><code>python main.py --input input_photos --output processed_photos
</code></pre>

<p>Или через Python:</p>

<pre><code>from portrait_enhancer import process_images
process_images("input_photos", "output_photos")
</code></pre>

<h2>🎯 Примеры работы</h2>

<div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="https://github.com/yourusername/smart-portrait-enhancer/raw/main/examples/before.jpg" width="300" alt="До обработки">
    <p><em>До обработки</em></p>
  </div>
  <div style="text-align: center;">
    <img src="https://github.com/yourusername/smart-portrait-enhancer/raw/main/examples/after.png" width="300" alt="После обработки">
    <p><em>После обработки</em></p>
  </div>
</div>

<h2>📝 Лицензия</h2>

<p>MIT License © 2023 Ваше Имя</p>

<h2>💡 Идеи для улучшения</h2>

<ul>
  <li>Добавить поддержку GPU для ускорения обработки</li>
  <li>Реализовать веб-интерфейс с Streamlit</li>
  <li>Добавить дополнительные параметры коррекции цвета</li>
</ul>

<p align="center">
  <a href="https://github.com/yourusername/smart-portrait-enhancer/issues">Сообщить об ошибке</a> •
  <a href="https://github.com/yourusername/smart-portrait-enhancer/pulls">Сделать вклад</a>
</p>
