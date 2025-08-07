using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;  // EF.Functions.ILike için
using stajprojem.Data;
using stajprojem.Models;
using System.Linq;

namespace stajprojem.Controllers
{
    public class FilteredProductsController : Controller
    {
        private readonly AppDbContext _context;

        public FilteredProductsController(AppDbContext context)
        {
            _context = context;
        }

        public IActionResult Index(int? CategoryId, string search)
        {
            // Kategorileri ViewBag ile gönderiyoruz (dropdown için)
            var categories = _context.Categories.ToList();
            ViewBag.Categories = categories;

            var products = _context.Products.AsQueryable();

            // Kategori filtresi varsa uygula
            if (CategoryId.HasValue && CategoryId.Value > 0)
                products = products.Where(p => p.category_id == CategoryId.Value);

            // Arama metni varsa, trimleyip büyük/küçük harf duyarsız arama yap
            if (!string.IsNullOrWhiteSpace(search))
            {
                var keyword = search.Trim();

                products = products.Where(p =>
                    p.product_name != null &&
                    EF.Functions.ILike(p.product_name, $"%{keyword}%"));
            }

            // Sonuçları liste olarak döndür
            return View(products.ToList());
        }
    }
}
