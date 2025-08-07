using Microsoft.AspNetCore.Mvc;
using stajprojem.Data; // AppDbContext burada
using stajprojem.Models;

namespace stajprojem.Controllers
{
    public class CategoriesController : Controller
    {
        private readonly AppDbContext _context;

        public CategoriesController(AppDbContext context)
        {
            _context = context;
        }

        public IActionResult Index()
        {
            var categories = _context.Categories.ToList();
            return View(categories);
        }
    }
}
