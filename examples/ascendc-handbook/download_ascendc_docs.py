#!/usr/bin/env python3
"""
AscendC 算子开发文档下载器
从昇腾社区官网下载完整的AscendC算子开发文档，保存为Markdown格式
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from html2text import HTML2Text
import html
import json

BASE_URL = "https://www.hiascend.com"
START_URL = "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0003.html"
OUTPUT_DIR = "/Volumes/GM9/code/Ascend-MLIR/examples/ascendc-handbook"

class AscendCDocDownloader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        self.visited_urls = set()
        self.pages = []
        self.h2t = HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = False
        self.h2t.body_width = 0
        self.h2t.unicode_snob = True
        self.h2t.skip_internal_links = False
        
    def get_page(self, url, retry=3):
        """获取页面内容"""
        for i in range(retry):
            try:
                print(f"  正在获取: {url}")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                response.encoding = 'utf-8'
                return response.text
            except Exception as e:
                print(f"  获取失败 (尝试 {i+1}/{retry}): {e}")
                if i < retry - 1:
                    time.sleep(2)
        return None
    
    def parse_page(self, html_content, url):
        """解析页面内容，提取标题、正文和下一篇链接"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        result = {
            'url': url,
            'title': '',
            'content': '',
            'next_url': None,
            'raw_html': html_content
        }
        
        title_tag = soup.find('title')
        if title_tag:
            result['title'] = title_tag.get_text(strip=True)
            if '-' in result['title']:
                result['title'] = result['title'].split('-')[0].strip()
        
        content_selectors = [
            {'class': 'document-content'},
            {'class': 'markdown-body'},
            {'class': 'content'},
            {'id': 'content'},
            {'class': 'article-content'},
            {'class': 'doc-content'},
        ]
        
        content_div = None
        for selector in content_selectors:
            content_div = soup.find('div', selector)
            if content_div:
                break
        
        if not content_div:
            main_tag = soup.find('main')
            if main_tag:
                content_div = main_tag
            else:
                article_tag = soup.find('article')
                if article_tag:
                    content_div = article_tag
        
        if content_div:
            for script in content_div.find_all('script'):
                script.decompose()
            for style in content_div.find_all('style'):
                style.decompose()
            for nav in content_div.find_all('nav'):
                nav.decompose()
            
            result['content'] = str(content_div)
        else:
            body = soup.find('body')
            if body:
                result['content'] = str(body)
        
        next_link = soup.find('a', {'class': re.compile(r'next|next-page|nextPage', re.I)})
        if next_link and next_link.get('href'):
            href = next_link.get('href')
            if href and not href.startswith('javascript'):
                result['next_url'] = urljoin(url, href)
        
        if not result['next_url']:
            next_link = soup.find('a', {'id': re.compile(r'next', re.I)})
            if next_link and next_link.get('href'):
                href = next_link.get('href')
                if href and not href.startswith('javascript'):
                    result['next_url'] = urljoin(url, href)
        
        if not result['next_url']:
            next_link = soup.find('a', {'title': re.compile(r'下一篇|下一页|Next', re.I)})
            if next_link and next_link.get('href'):
                href = next_link.get('href')
                if href and not href.startswith('javascript'):
                    result['next_url'] = urljoin(url, href)
        
        if not result['next_url']:
            all_links = soup.find_all('a', href=True)
            for link in all_links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                if '下一篇' in text or '下一页' in text or 'next' in text.lower():
                    if href and not href.startswith('javascript'):
                        result['next_url'] = urljoin(url, href)
                        break
        
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
        print("AscendC 算子开发文档下载器")
        print("=" * 60)
        
        current_url = START_URL
        page_num = 0
        max_pages = 500
        
        while current_url and page_num < max_pages:
            if current_url in self.visited_urls:
                print(f"  检测到重复URL，停止下载")
                break
            
            self.visited_urls.add(current_url)
            page_num += 1
            
            print(f"\n[{page_num}] 处理页面...")
            
            html_content = self.get_page(current_url)
            if not html_content:
                print(f"  无法获取页面内容，跳过")
                time.sleep(1)
                continue
            
            page_data = self.parse_page(html_content, current_url)
            self.pages.append(page_data)
            
            print(f"  标题: {page_data['title']}")
            print(f"  内容长度: {len(page_data['content'])} 字符")
            
            if page_data['next_url']:
                print(f"  下一篇: {page_data['next_url']}")
                current_url = page_data['next_url']
            else:
                print(f"  没有找到下一篇链接，下载完成")
                break
            
            time.sleep(1)
        
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
                    'next_url': page['next_url']
                }
                for i, page in enumerate(self.pages, 1)
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n元数据保存到: {filepath}")


def main():
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


if __name__ == "__main__":
    main()
