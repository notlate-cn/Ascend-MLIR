#!/usr/bin/env python3
"""
AscendC 算子开发文档下载器 (Selenium版 - 索引模式)
从昇腾社区官网下载完整的AscendC算子开发文档，保存为Markdown格式
策略：先从索引页提取所有链接，再逐个下载
"""

import os
import re
import time
import json
import html
from urllib.parse import urljoin, urlparse
from html2text import HTML2Text

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL = "https://www.hiascend.com"
START_URL = "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0003.html"
OUTPUT_DIR = "/Volumes/GM9/code/Ascend-MLIR/examples/ascendc-handbook"


class AscendCDocDownloader:
    def __init__(self):
        self.driver = self._init_driver()
        self.visited_urls = set()
        self.pages = []
        self.h2t = HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = False
        self.h2t.body_width = 0
        self.h2t.unicode_snob = True
        self.h2t.skip_internal_links = False
        
    def _init_driver(self):
        """初始化Selenium WebDriver"""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(60)
        return driver
    
    def get_page_content(self, url, wait_class="the-article-body"):
        """获取页面内容"""
        try:
            print(f"  正在加载: {url}")
            self.driver.get(url)
            
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.CLASS_NAME, wait_class))
            )
            
            time.sleep(1)
            
            return True
        except TimeoutException:
            print(f"  页面加载超时")
            return False
        except Exception as e:
            print(f"  加载失败: {e}")
            return False
    
    def extract_all_links_from_index(self):
        """从索引页提取所有API文档链接"""
        print("\n正在从索引页提取所有文档链接...")
        
        links = set()
        
        try:
            all_links = self.driver.find_elements(By.TAG_NAME, "a")
            
            for link in all_links:
                try:
                    href = link.get_attribute('href')
                    if href and 'atlasascendc_api_07_' in href and href.endswith('.html'):
                        if 'atlasascendc_api_07_0003.html' not in href:
                            links.add(href)
                except:
                    continue
            
            print(f"  从索引页提取到 {len(links)} 个链接")
            
        except Exception as e:
            print(f"  提取链接时出错: {e}")
        
        return sorted(list(links), key=lambda x: self._extract_page_number(x))
    
    def _extract_page_number(self, url):
        """从URL中提取页码用于排序"""
        match = re.search(r'atlasascendc_api_07_(\d+)\.html', url)
        if match:
            return int(match.group(1))
        return 0
    
    def parse_current_page(self, url):
        """解析当前页面"""
        result = {
            'url': url,
            'title': '',
            'content': '',
        }
        
        try:
            result['title'] = self.driver.title
            if '-' in result['title']:
                result['title'] = result['title'].split('-')[0].strip()
            
            try:
                article_body = self.driver.find_element(By.CLASS_NAME, "the-article-body")
                result['content'] = article_body.get_attribute('outerHTML')
            except NoSuchElementException:
                try:
                    article = self.driver.find_element(By.CLASS_NAME, "article-content")
                    result['content'] = article.get_attribute('outerHTML')
                except NoSuchElementException:
                    result['content'] = self.driver.find_element(By.TAG_NAME, "body").get_attribute('innerHTML')
            
        except Exception as e:
            print(f"  解析页面时出错: {e}")
        
        return result
    
    def html_to_markdown(self, html_content, title):
        """将HTML转换为Markdown"""
        try:
            md_content = self.h2t.handle(html_content)
            md_content = f"# {title}\n\n{md_content}"
            return md_content
        except Exception as e:
            print(f"  Markdown转换失败: {e}")
            return f"# {title}\n\n[内容转换失败，请查看原始HTML]"
    
    def sanitize_filename(self, filename):
        """清理文件名"""
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'\s+', '_', filename)
        filename = filename[:100]
        return filename
    
    def download_all(self):
        """下载所有文档页面"""
        print("=" * 60)
        print("AscendC 算子开发文档下载器 (索引模式)")
        print("=" * 60)
        
        print("\n[步骤1] 加载索引页...")
        if not self.get_page_content(START_URL):
            print("无法加载索引页，退出")
            return []
        
        print("\n[步骤2] 提取所有文档链接...")
        all_links = self.extract_all_links_from_index()
        
        if not all_links:
            print("未找到任何文档链接，退出")
            return []
        
        print(f"\n找到 {len(all_links)} 个文档页面待下载")
        
        print("\n[步骤3] 下载索引页...")
        index_page = self.parse_current_page(START_URL)
        self.pages.append(index_page)
        print(f"  标题: {index_page['title']}")
        
        print("\n[步骤4] 逐个下载文档页面...")
        total = len(all_links)
        consecutive_failures = 0
        
        for i, url in enumerate(all_links, 1):
            if url in self.visited_urls:
                continue
            
            self.visited_urls.add(url)
            
            print(f"\n[{i}/{total}] 处理: {url}")
            
            if not self.get_page_content(url):
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    print(f"  连续失败{consecutive_failures}次，暂停下载")
                    time.sleep(10)
                    consecutive_failures = 0
                continue
            
            consecutive_failures = 0
            
            page_data = self.parse_current_page(url)
            self.pages.append(page_data)
            
            print(f"  标题: {page_data['title']}")
            print(f"  内容长度: {len(page_data['content'])} 字符")
            
            time.sleep(0.5)
        
        print(f"\n共下载 {len(self.pages)} 个页面")
        return self.pages
    
    def save_as_markdown(self):
        """保存为单独的Markdown文件"""
        md_dir = os.path.join(OUTPUT_DIR, "markdown")
        os.makedirs(md_dir, exist_ok=True)
        
        print(f"\n保存Markdown文件到: {md_dir}")
        
        for i, page in enumerate(self.pages, 1):
            title = page['title'] or f"页面_{i}"
            filename = f"{i:03d}_{self.sanitize_filename(title)}.md"
            filepath = os.path.join(md_dir, filename)
            
            md_content = self.html_to_markdown(page['content'], title)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"<!-- 原始URL: {page['url']} -->\n")
                f.write(f"<!-- 下载时间: {time.strftime('%Y-%m-%d %H:%M:%S')} -->\n\n")
                f.write(md_content)
            
            print(f"  保存: {filename}")
        
        print(f"Markdown文件保存完成")
    
    def save_as_combined_markdown(self):
        """保存为合并的Markdown文件"""
        filepath = os.path.join(OUTPUT_DIR, "AscendC_算子开发文档_完整版.md")
        
        print(f"\n保存合并Markdown文件: {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# AscendC 算子开发文档\n\n")
            f.write("> 来源: 昇腾社区官网\n")
            f.write(f"> 下载时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"> 页面数量: {len(self.pages)}\n\n")
            f.write("---\n\n")
            f.write("## 目录\n\n")
            
            for i, page in enumerate(self.pages, 1):
                title = page['title'] or f"页面_{i}"
                anchor = self.sanitize_filename(title).lower()
                f.write(f"{i}. [{title}](#{anchor})\n")
            
            f.write("\n---\n\n")
            
            for i, page in enumerate(self.pages, 1):
                title = page['title'] or f"页面_{i}"
                f.write(f"\n\n---\n\n")
                f.write(f"<!-- 第 {i}/{len(self.pages)} 页 -->\n")
                f.write(f"<!-- 原始URL: {page['url']} -->\n\n")
                
                md_content = self.html_to_markdown(page['content'], title)
                f.write(md_content)
        
        print(f"合并Markdown文件保存完成")
    
    def save_as_html(self):
        """保存为HTML文件"""
        html_dir = os.path.join(OUTPUT_DIR, "html")
        os.makedirs(html_dir, exist_ok=True)
        
        print(f"\n保存HTML文件到: {html_dir}")
        
        for i, page in enumerate(self.pages, 1):
            title = page['title'] or f"页面_{i}"
            filename = f"{i:03d}_{self.sanitize_filename(title)}.html"
            filepath = os.path.join(html_dir, filename)
            
            html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        pre {{ background: #f4f4f4; padding: 15px; overflow-x: auto; border-radius: 5px; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f4f4f4; }}
        img {{ max-width: 100%; }}
    </style>
</head>
<body>
    <p><small>原始URL: <a href="{page['url']}">{page['url']}</a></small></p>
    <hr>
    {page['content']}
</body>
</html>"""
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"  保存: {filename}")
        
        print(f"HTML文件保存完成")
    
    def save_metadata(self):
        """保存元数据"""
        filepath = os.path.join(OUTPUT_DIR, "metadata.json")
        
        metadata = {
            'download_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_pages': len(self.pages),
            'start_url': START_URL,
            'pages': [
                {
                    'index': i,
                    'url': page['url'],
                    'title': page['title'],
                }
                for i, page in enumerate(self.pages, 1)
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n元数据保存到: {filepath}")
    
    def close(self):
        """关闭浏览器"""
        if self.driver:
            self.driver.quit()


def main():
    downloader = None
    try:
        downloader = AscendCDocDownloader()
        downloader.download_all()
        
        if downloader.pages:
            downloader.save_as_markdown()
            downloader.save_as_combined_markdown()
            downloader.save_as_html()
            downloader.save_metadata()
            
            print("\n" + "=" * 60)
            print("下载完成!")
            print("=" * 60)
            print(f"输出目录: {OUTPUT_DIR}")
            print(f"总页数: {len(downloader.pages)}")
            print("\n文件说明:")
            print("  - markdown/       : 单独的Markdown文件")
            print("  - AscendC_算子开发文档_完整版.md : 合并的完整文档")
            print("  - html/           : HTML格式文件")
            print("  - metadata.json   : 下载元数据")
        else:
            print("\n下载失败，未获取到任何页面")
    except KeyboardInterrupt:
        print("\n用户中断下载")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if downloader:
            downloader.close()


if __name__ == "__main__":
    main()
