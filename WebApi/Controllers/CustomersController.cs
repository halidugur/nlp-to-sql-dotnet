using Microsoft.AspNetCore.Mvc;
using stajprojem.Data;       // AppDbContext için
using stajprojem.Models;     // Customer modeli için
using System.Linq;           // ToList() ve LINQ için
using System.Threading.Tasks;

namespace stajprojem.Controllers
{
    public class CustomersController : Controller
    {
        private readonly AppDbContext _context;

        public CustomersController(AppDbContext context)
        {
            _context = context;
        }

        public IActionResult Index()
        {
            var customers = _context.Customers.ToList();
            // Dikkat: "Customers" ile büyük harfle!
            return View(customers);
        }
    }
}
